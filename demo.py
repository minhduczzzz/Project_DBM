import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import streamlit as st
from dotenv import load_dotenv
from model import DogBreedResNet
from database import ChatDatabase
from gemini_chatbot import GeminiChatbot

# Load environment variables
load_dotenv()

# DEBUG: Check if .env was loaded
api_key_loaded = os.getenv("GEMINI_API_KEY")
if api_key_loaded:
    print(f"[DEBUG] ✅ GEMINI_API_KEY loaded from .env: {api_key_loaded[:20]}...{api_key_loaded[-5:]}")
else:
    print("[DEBUG] ❌ GEMINI_API_KEY not found in environment")


# --- Setup Page Config ---
st.set_page_config(
    page_title="Dog Breed Classifier",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Session State ---
if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = None
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "db" not in st.session_state:
    st.session_state.db = ChatDatabase()
if "api_key_valid" not in st.session_state:
    st.session_state.api_key_valid = False
if "current_breed" not in st.session_state:
    st.session_state.current_breed = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# --- Custom CSS for styling ---
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .title {
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        color: #1f77d4;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
    }
    .breed-name {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77d4;
    }
    .confidence {
        font-size: 1.3rem;
        color: #2ca02c;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained ResNet model"""
    checkpoint_path = "./training_models/best_resnet.pth"
    
    if not os.path.exists(checkpoint_path):
        st.error(f"""
        ❌ **Model checkpoint not found!**
        
        Expected path: `{checkpoint_path}`
        
        **Solution:**
        1. Make sure you have trained the model using `python train_cnn.py`
        2. The training script will create the checkpoint automatically
        3. Ensure the `training_models/` folder exists and contains `best_resnet.pth`
        """)
        st.stop()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Check if checkpoint has required keys
        if "class_to_idx" not in checkpoint:
            st.error(f"""
            ❌ **Invalid model checkpoint!**
            
            The checkpoint file is missing the 'class_to_idx' key.
            This usually means the checkpoint was created with an older version.
            
            **Solution:**
            1. Delete the old checkpoint file: `training_models/best_resnet.pth`
            2. Run the training script again: `python train_cnn.py`
            3. This will create a new valid checkpoint with all required data
            """)
            st.stop()
        
        # Extract class mappings from checkpoint
        class_to_idx = checkpoint["class_to_idx"]
        idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        num_classes = len(class_to_idx)
        
        # Load model architecture
        model = DogBreedResNet(num_classes=num_classes, pretrained=False).to(device)
        
        if "model" not in checkpoint:
            st.error("❌ Error: 'model' key not found in checkpoint")
            st.stop()
        
        model.load_state_dict(checkpoint["model"])
        model.eval()
        
        return model, class_to_idx, idx_to_class, device
    
    except KeyError as e:
        st.error(f"""
        ❌ **Checkpoint key error: {str(e)}**
        
        The checkpoint file is damaged or incomplete.
        
        **Solution:**
        1. Delete the checkpoint: `training_models/best_resnet.pth`
        2. Retrain the model: `python train_cnn.py`
        """)
        st.stop()
    except Exception as e:
        st.error(f"""
        ❌ **Unexpected error loading model:**
        
        `{str(e)}`
        
        **Troubleshooting:**
        - Check if PyTorch and dependencies are properly installed
        - Try deleting the checkpoint and retraining: `python train_cnn.py`
        - Check the Python version compatibility
        """)
        st.stop()


def preprocess_image(image: Image.Image, device: torch.device) -> torch.Tensor:
    """Preprocess the image for model input"""
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], 
                  std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor


def predict_breed(image: Image.Image, model, idx_to_class: dict, device: torch.device) -> tuple:
    """Predict dog breed from image"""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess image
        image_tensor = preprocess_image(image, device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, dim=0)
            
            predicted_breed = idx_to_class[predicted_idx.item()]
            confidence_score = confidence.item() * 100
            
            # Get top 3 predictions
            top_probs, top_indices = torch.topk(probabilities, k=3)
            top_predictions = [
                (idx_to_class[idx.item()], prob.item() * 100)
                for idx, prob in zip(top_indices, top_probs)
            ]
        
        return predicted_breed, confidence_score, top_predictions
    
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        return None, None, None


# --- Main App ---
st.markdown('<div class="title">🐕 Dog Breed Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Dog Breed Recognition using ResNet18</div>', unsafe_allow_html=True)

# Load model
model, class_to_idx, idx_to_class, device = load_model()

# Two column layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📸 Upload Image")
    
    # Image upload option
    uploaded_file = st.file_uploader(
        "Choose a dog image (PNG, JPG, JPEG, BMP):",
        type=["png", "jpg", "jpeg", "bmp"],
        help="Upload a clear image of a dog for best results"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True, caption="Uploaded Image")

with col2:
    st.header("📊 Prediction Results")
    
    if uploaded_file is not None:
        with st.spinner("🔍 Analyzing image..."):
            predicted_breed, confidence, top_preds = predict_breed(
                image, model, idx_to_class, device
            )
        
        if predicted_breed:
            # Format breed name nicely
            display_breed = predicted_breed.replace("_", " ").title()
            
            # Main prediction result
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown(f'<div class="breed-name">🐶 {display_breed}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="confidence">Confidence: {confidence:.2f}%</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Top 3 predictions
            st.subheader("🏆 Top 3 Predictions")
            for i, (breed, conf) in enumerate(top_preds, 1):
                breed_display = breed.replace("_", " ").title()
                st.write(f"{i}. **{breed_display}** - {conf:.2f}%")
                st.progress(conf / 100)
            
            # Initialize chat for this breed
            if st.session_state.current_breed != predicted_breed:
                st.session_state.current_breed = predicted_breed
                st.session_state.chat_session_id = st.session_state.db.create_session(
                    predicted_breed, confidence
                )
                st.session_state.chat_messages = []
                
                # Initialize chatbot for this breed if API key is valid
                if st.session_state.api_key_valid and st.session_state.chatbot:
                    st.session_state.chatbot.start_conversation(predicted_breed)
    
    else:
        st.info("👆 Upload an image to see predictions")

# --- CHATBOT SECTION ---
if st.session_state.current_breed and st.session_state.api_key_valid:
    st.divider()
    breed_display = st.session_state.current_breed.replace("_", " ").title()
    st.header("💬 Ask About This Breed")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_messages:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])
    
    # Chat input
    user_input = st.chat_input(f"Ask about {breed_display}...")
    
    if user_input:
        # Add user message to display
        st.chat_message("user").write(user_input)
        
        with st.spinner("🤔 Thinking..."):
            try:
                # Get response from Gemini
                response = st.session_state.chatbot.chat(user_input)
                
                # Display assistant response
                st.chat_message("assistant").write(response)
                
                # Save to chat history (display)
                st.session_state.chat_messages.append({"role": "user", "content": user_input})
                st.session_state.chat_messages.append({"role": "assistant", "content": response})
                
                # Save to database
                st.session_state.db.save_message(
                    st.session_state.chat_session_id, "user", user_input
                )
                st.session_state.db.save_message(
                    st.session_state.chat_session_id, "assistant", response
                )
                
                # Rerun to update chat display
                st.rerun()
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
elif st.session_state.current_breed and not st.session_state.api_key_valid:
    st.divider()
    st.warning("⚠️ Please set up Gemini API key in the sidebar to use the chatbot")


# Sidebar information
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This application uses a **ResNet18** deep learning model trained on dog breed images 
    to classify over 120 different dog breeds.
    
    **How to use:**
    1. Upload a clear image of a dog
    2. The AI will analyze the image
    3. View the predicted breed and confidence score
    4. Chat with the breed specialist chatbot
    
    **Model Details:**
    - Architecture: ResNet18
    - Number of Classes: 120 dog breeds
    - Input Size: 224 × 224 pixels
    - Framework: PyTorch
    """)
    
    st.divider()
    
    # API Key Setup
    st.header("🔑 Gemini API Setup")
    st.write("""
    Get your **FREE** Gemini API key here:
    
    https://makersuite.google.com/app/apikey
    """)
    
    api_key_input = st.text_input(
        "Enter Gemini API Key:",
        type="password",
        help="Your API key will not be saved"
    )
    
    if api_key_input:
        try:
            st.session_state.chatbot = GeminiChatbot(api_key=api_key_input)
            st.session_state.api_key_valid = True
            # If breed already detected, initialize conversation
            if st.session_state.current_breed:
                st.session_state.chatbot.start_conversation(st.session_state.current_breed)
            st.success("✅ API Key Valid!")
        except Exception as e:
            st.error(f"❌ Invalid API Key")
            st.session_state.api_key_valid = False
    
    # Check for environment variable
    if not api_key_input and os.getenv("GEMINI_API_KEY"):
        try:
            st.session_state.chatbot = GeminiChatbot()
            st.session_state.api_key_valid = True
            # If breed already detected, initialize conversation
            if st.session_state.current_breed:
                st.session_state.chatbot.start_conversation(st.session_state.current_breed)
            st.success("✅ Using API Key from Environment")
        except:
            st.session_state.api_key_valid = False
    
    st.divider()
    
    st.header("⚙️ System Info")
    if torch.cuda.is_available():
        st.success(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("⚠️ Using CPU (slower predictions)")
    
    st.write(f"**PyTorch Version:** {torch.__version__}")
    st.write(f"**Device:** {device}")
    st.write(f"**Number of Classes:** {len(idx_to_class)}")
    
    st.divider()
    
    # Chat History
    st.header("📚 Chat History")
    
    all_sessions = st.session_state.db.get_all_sessions()
    
    if all_sessions:
        st.write(f"**Total Sessions:** {len(all_sessions)}")
        
        for session in all_sessions[:5]:  # Show last 5
            with st.expander(f"🐕 {session['breed_name']} - {session['created_at'][:10]}"):
                st.write(f"**Confidence:** {session['confidence']:.2f}%")
                messages = st.session_state.db.get_session_history(session['session_id'])
                st.write(f"**Messages:** {len(messages)}")
                
                if st.button(f"Delete", key=f"del_{session['session_id']}"):
                    st.session_state.db.delete_session(session['session_id'])
                    st.rerun()
    else:
        st.info("No chat history yet. Start a conversation!")
