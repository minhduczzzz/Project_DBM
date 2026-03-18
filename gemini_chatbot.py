import os
from typing import List, Dict, Optional
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class GeminiChatbot:
    """Chatbot powered by Google Gemini API"""
    
    def __init__(self, api_key: str = None, breed_name: str = None):
        """
        Initialize Gemini chatbot
        
        Args:
            api_key: Google Gemini API key. If None, will look for GEMINI_API_KEY env var
            breed_name: Dog breed name (optional). If provided, automatically initializes conversation
        """
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError(
                "❌ API key not found!\n"
                "Please set GEMINI_API_KEY environment variable or pass it as parameter.\n"
                "Get your free API key at: https://makersuite.google.com/app/apikey"
            )
        
        # DEBUG: Print API key info
        print(f"[DEBUG] API Key Found: {api_key[:20]}...{api_key[-5:]}")
        
        genai.configure(api_key=api_key)
        
        # DEBUG: List available models
        print("[DEBUG] Fetching available models...")
        try:
            models = genai.list_models()
            available_models = []
            for m in models:
                print(f"[DEBUG] Available: {m.name} - Supports: {m.supported_generation_methods}")
                available_models.append(m.name)
            print(f"[DEBUG] Total available models: {len(available_models)}")
        except Exception as e:
            print(f"[DEBUG] Error listing models: {e}")
            available_models = []
        
        # Try to find a working model from available models
        self.model = None
        self.model_name = None
        
        if available_models:
            print("\n[DEBUG] Trying available models from API...")
            for m in available_models:
                try:
                    print(f"[DEBUG] Testing model: {m}")
                    self.model = genai.GenerativeModel(m)
                    # Test with a simple call
                    response = self.model.generate_content("Hi")
                    print(f"[DEBUG] ✅ Model '{m}' works!")
                    self.model_name = m
                    break
                except Exception as e:
                    print(f"[DEBUG] ❌ Model '{m}' failed: {str(e)[:60]}")
                    continue
        else:
            # Fallback: try hardcoded models
            print("\n[DEBUG] No models from API, trying hardcoded list...")
            model_names = [
                'gemini-2.0-flash',
                'gemini-2.0-flash-latest',
                'gemini-1.5-flash-latest',
                'gemini-1.5-flash',
                'gemini-1.5-pro',
                'gemini-pro',
                'gemini-pro-vision',
            ]
            
            for model_name in model_names:
                try:
                    print(f"[DEBUG] Trying model: {model_name}")
                    self.model = genai.GenerativeModel(model_name)
                    response = self.model.generate_content("Hi")
                    print(f"[DEBUG] ✅ Model '{model_name}' works!")
                    self.model_name = model_name
                    break
                except Exception as e:
                    print(f"[DEBUG] ❌ Model '{model_name}' failed: {str(e)[:60]}")
                    continue
        
        if not self.model:
            raise ValueError(
                "❌ No working models found!\n"
                "Your API key may be invalid or restricted.\n\n"
                "Solutions:\n"
                "1. Create a NEW API key: https://makersuite.google.com/app/apikey\n"
                "2. Make sure API is ENABLED in Google Cloud\n"
                "3. Check if you have quota available\n"
                "4. Try with a different Google account"
            )
        
        self.conversation_history = []
        self.system_prompt = None
        
        # If breed_name is provided, start conversation immediately
        if breed_name:
            self.start_conversation(breed_name)
    
    def create_system_prompt(self, breed_name: str) -> str:
        """Create a system prompt for the breed specialist chatbot"""
        breed_display = breed_name.replace("_", " ").title()
        
        system_prompt = f"""You are an expert on {breed_display} dogs. You provide authoritative, detailed, and helpful information about this breed.

When answering questions about {breed_display} dogs, consider these aspects:
- Physical characteristics and appearance
- Temperament and personality traits
- Health issues and care requirements
- Training difficulty and techniques
- Exercise and activity needs
- Grooming and maintenance
- Suitability for different living situations
- History and origin of the breed
- Average lifespan and size
- Cost and availability
- Famous {breed_display} dogs

Always:
1. Provide accurate and verified information
2. Be enthusiastic about {breed_display} dogs
3. Suggest when professional veterinary advice is needed
4. Answer in a friendly, conversational manner
5. Use relevant facts and examples
6. Correct any misconceptions about {breed_display} dogs

If asked about other dog breeds, politely redirect the conversation back to {breed_display} while acknowledging the question."""
        
        return system_prompt
    
    def start_conversation(self, breed_name: str):
        """Initialize a new conversation with the breed specialist"""
        self.conversation_history = []
        self.system_prompt = self.create_system_prompt(breed_name)
    
    def chat(self, user_message: str) -> str:
        """
        Send a message and get a response
        
        Args:
            user_message: User's question or statement
            
        Returns:
            Chatbot's response
        """
        if not self.system_prompt:
            return "❌ Error: Chatbot not initialized. Please call start_conversation() first with breed name."
        
        print(f"[DEBUG] Using model: {self.model_name}")
        print(f"[DEBUG] User message: {user_message[:50]}...")
        
        # Add system context to the beginning of conversation
        if not self.conversation_history:
            messages = [
                {"role": "user", "parts": f"{self.system_prompt}\n\nUser: {user_message}"}
            ]
        else:
            messages = self.conversation_history + [
                {"role": "user", "parts": user_message}
            ]
        
        try:
            # Create chat session with the history
            chat = self.model.start_chat(history=self._convert_history_format(messages[:-1]))
            response = chat.send_message(messages[-1]["parts"])
            
            print(f"[DEBUG] Response received: {len(response.text)} characters")
            
            # Store in conversation history
            self.conversation_history.append({"role": "user", "parts": user_message})
            self.conversation_history.append({"role": "model", "parts": response.text})
            
            return response.text
        
        except Exception as e:
            print(f"[DEBUG] Error in chat: {str(e)}")
            return f"❌ Error: {str(e)}\n\nMake sure your Gemini API key is valid."
    
    def _convert_history_format(self, history: List[Dict]) -> List:
        """Convert chat history to Gemini API format"""
        gemini_history = []
        for msg in history:
            gemini_history.append({
                "role": msg["role"],
                "parts": [{"text": msg["parts"]}]
            })
        return gemini_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history
