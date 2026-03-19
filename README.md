# 🐕 Dog Breed Classifier - Project DBM302m Team 5

AI-powered dog breed recognition using ResNet18 deep learning model with intelligent chatbot.

## 📋 Project Overview

This project implements a dog breed classifier that can predict the breed of a dog from an image. It includes:
- **Model Training**: ResNet18 architecture trained on 120 dog breed classes
- **Streamlit Demo**: Web-based user interface with integrated chatbot
- **Smart Chatbot**: AI-powered conversations powered by Google Gemini
- **Chat History**: SQLite database to store all conversations
- **Desktop App**: CustomTkinter GUI with Wikipedia integration
- **Dataset Support**: Handles dog breed images with breed labels

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Demo with Chatbot

```bash
streamlit run demo.py
```

This will open the web app at `http://localhost:8501`

**Features:**
- 📸 Upload dog images (PNG, JPG, JPEG, BMP)
- 🔍 Get instant breed predictions with confidence scores
- 🏆 View top 3 predictions with probability bars
- 💬 **NEW: Chat with AI breed specialist** (requires Gemini API key)
- 📚 **NEW: Automatic chat history with SQLite**
- ⚙️ Real-time GPU/CPU detection

### 3. Setup Chatbot (Optional but Recommended)

To enable the intelligent chatbot feature:

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_actual_key_here
   ```

3. Get a **FREE** Gemini API key: https://makersuite.google.com/app/apikey

4. Run the app - it will automatically load the API key:
   ```bash
   streamlit run demo.py
   ```

**Note:** `.env` file is automatically ignored by Git for security.

👉 **See [CHATBOT_SETUP.md](CHATBOT_SETUP.md) for detailed instructions**

### 4. Run the Desktop App (Optional)

If you prefer a desktop interface:

```bash
python app.py
```

This version includes Wikipedia integration to fetch breed information automatically.

## 📁 Project Structure

```
Project_AIL-_team_3/
├── demo.py                  # Streamlit web demo with chatbot
├── app.py                   # Desktop GUI app (CustomTkinter)
├── model.py                 # ResNet18 model architecture
├── train_cnn.py             # Training script
├── dataset.py               # Dataset classes
├── gemini_chatbot.py        # Chatbot with Gemini API
├── database.py              # SQLite database for chat history
├── labels.csv               # Dog breed labels
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variables template
├── .env                     # Environment variables (local use only - ignored by git)
├── .gitignore               # Git ignore rules
├── CHATBOT_SETUP.md         # Chatbot setup guide
├── README.md                # This file
├── chat_history.db          # SQLite chat database (auto-created)
├── train/                   # Training images directory
└── training_models/         # Saved model checkpoints
    └── best_resnet.pth      # Best trained model
```

## 📦 Requirements

- Python 3.8 or higher
- PyTorch 2.0+
- Streamlit 1.28+
- Google Generative AI library (for chatbot)
- All packages listed in `requirements.txt`

## 🎯 Key Features

### Streamlit Demo (demo.py)
- 🖼️ Modern web interface
- 📸 Easy image upload
- 📊 Real-time predictions with confidence scores
- 🏆 Top 3 breed predictions with probability visualization
- ⚙️ System info (GPU/CPU detection)
- 📱 Responsive design for desktop and mobile

### 💬 Intelligent Chatbot Features (NEW!)
- 🤖 AI-powered breed specialist chatbot
- 🎯 Custom system prompt for each breed
- 📝 Multi-turn conversations
- 💾 Automatic chat history saving
- 🔍 View past conversations
- 🗑️ Delete conversation history
- 🔐 Secure API key handling

### Desktop App (app.py)
- 🎨 Modern GUI with CustomTkinter
- 🌐 Wikipedia integration for breed information
- 🧵 Multi-threaded processing (non-blocking UI)
- 🇻🇳 Vietnamese and English support

## 🤖 Chatbot System Prompt

The chatbot automatically becomes a specialist for each detected breed:

```
You are an expert on [BREED_NAME] dogs. You provide:
- Physical characteristics and appearance
- Temperament and personality traits
- Health issues and care requirements
- Training difficulty and techniques
- Exercise and activity needs
- Grooming and maintenance
- Suitability for different living situations
- History and origin of the breed
- Cost, availability, and lifespan
```

## 🔧 Model Information

- **Architecture**: ResNet18 (Residual Neural Network)
- **Number of Classes**: 120 dog breeds
- **Input Size**: 224 × 224 pixels
- **Framework**: PyTorch
- **Pre-training**: ImageNet weights
- **Training Data Augmentation**: 
  - Random horizontal flip
  - Random rotation (±20°)
  - Color jitter
  - Normalization with ImageNet stats

## 📊 Training

To train the model from scratch:

```bash
python train_cnn.py
```

Configuration:
- **Epochs**: 50
- **Batch Size**: 32
- **Learning Rate**: 1e-4
- **Optimizer**: Adam with L2 regularization
- **Loss Function**: CrossEntropyLoss
- **Early Stopping**: Patience of 10 epochs
- **Scheduler**: ReduceLROnPlateau

## 💡 Usage Tips

### General Usage
1. Use clear, well-lit images of dogs
2. Ensure the dog's face/body is visible
3. Avoid heavily cropped or blurry images
4. Single dog per image works best

### ChatBot Tips
1. Ask specific questions about the breed
2. The bot remembers your entire conversation
3. All chats are saved automatically
4. Use natural language (Vietnamese or English)

### GPU Acceleration
The app automatically detects and uses GPU if available. Training is significantly faster with CUDA-enabled GPU.

## 💾 Chat Database (SQLite)

The app automatically creates `chat_history.db` with:
- **chat_sessions**: Session info (breed, confidence, timestamp)
- **chat_messages**: Individual messages (role, content, timestamp)

View using any SQLite viewer or:
```python
import sqlite3
conn = sqlite3.connect("chat_history.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM chat_sessions")
print(cursor.fetchall())
```

## 🔍 Troubleshooting

### Model checkpoint not found
```bash
python train_cnn.py  # Train the model first
```

### Slow predictions on CPU
GPU acceleration is recommended. If unavailable, predictions will be slower but functional.

### Chatbot API Key errors
- Get key at: https://makersuite.google.com/app/apikey
- Set environment variable: `GEMINI_API_KEY=your_key`
- Or input directly in the sidebar

### Module import errors
```bash
pip install -r requirements.txt --upgrade
```

### Chat not saving
- Ensure folder has write permissions
- Check `chat_history.db` size
- Delete and recreate database if corrupted

## 📚 Additional Resources

- [Chatbot Setup Guide](CHATBOT_SETUP.md)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [Streamlit Documentation](https://docs.streamlit.io)
- [PyTorch Documentation](https://pytorch.org/docs)

## 👥 Team

Project DBM302m - Team 5 (FPT University)

## 📄 License

This project is for educational purposes.
├── model.py                 # ResNet18 model architecture
├── train_cnn.py             # Training script
├── dataset.py               # Dataset classes
├── labels.csv               # Dog breed labels
├── requirements.txt         # Python dependencies
├── train/                   # Training images directory
└── training_models/         # Saved model checkpoints
    └── best_resnet.pth      # Best trained model
```

## 📦 Requirements

- Python 3.8 or higher
- PyTorch 2.0+
- Streamlit 1.28+
- All packages listed in `requirements.txt`

## 🎯 Key Features

### Streamlit Demo (demo.py)
- 💻 Modern web interface
- 📸 Easy image upload
- 📊 Real-time predictions with confidence scores
- 🏆 Top 3 breed predictions with probability visualization
- ⚙️ System info (GPU/CPU detection)
- 📱 Responsive design for desktop and mobile

### Desktop App (app.py)
- 🎨 Dark mode GUI with CustomTkinter
- 🌐 Wikipedia integration for breed information
- 🧵 Multi-threaded processing (non-blocking UI)
- 🎭 Vietnamese and English support

## 🔧 Model Information

- **Architecture**: ResNet18 (Residual Neural Network)
- **Number of Classes**: 120 dog breeds
- **Input Size**: 224 × 224 pixels
- **Framework**: PyTorch
- **Pre-training**: ImageNet weights
- **Training Data Augmentation**: 
  - Random horizontal flip
  - Random rotation (±20°)
  - Color jitter
  - Normalization with ImageNet stats

## 📊 Training

To train the model from scratch:

```bash
python train_cnn.py
```

Configuration:
- **Epochs**: 50
- **Batch Size**: 32
- **Learning Rate**: 1e-4
- **Optimizer**: Adam with L2 regularization
- **Loss Function**: CrossEntropyLoss
- **Early Stopping**: Patience of 10 epochs
- **Scheduler**: ReduceLROnPlateau

## 💡 Usage Tips

### For Best Results:
1. Use clear, well-lit images of dogs
2. Ensure the dog's face/body is visible
3. Avoid heavily cropped or blurry images
4. Single dog per image works best

### GPU Acceleration:
The app automatically detects and uses GPU if available. Training is significantly faster with CUDA-enabled GPU.

## 🔍 Troubleshooting

### Model checkpoint not found
Make sure you have trained the model or downloaded `best_resnet.pth` and placed it in the `training_models/` folder.

### Slow predictions on CPU
GPU acceleration is recommended. If GPU is not available, predictions will be slower but still functional.

### Module import errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

## 📝 Notes

- The model was trained on the Stanford Dogs Dataset with 120 dog breeds
- Top-k accuracy is used for evaluation
- Real-time predictions take 1-3 seconds depending on hardware
- The model can be easily fine-tuned for custom dog breed datasets

## 👥 Team

Project DBM302m - Team 5 (FPT University)

## 📄 License

This project is for educational purposes.
