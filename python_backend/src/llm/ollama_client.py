import os
import requests
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

class OllamaClient:
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("MODEL_NAME", "llama2")
        self.conversation_history: List[Dict[str, str]] = []

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response using the Ollama API
        """
        messages = self.conversation_history.copy()
        
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                }
            )
            response.raise_for_status()
            
            assistant_message = response.json()["message"]["content"]
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
            
        except requests.exceptions.RequestException as e:
            print(f"Error generating response: {e}")
            return ""

    def set_model(self, model_name: str) -> None:
        """
        Change the model being used
        """
        self.model = model_name

    def clear_history(self) -> None:
        """
        Clear the conversation history
        """
        self.conversation_history = []

    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history
        """
        return self.conversation_history
