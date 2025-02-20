from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv
import ollama

load_dotenv()

class LLMManager:
    def __init__(self):
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_client = ollama.Client(host=ollama_url)
        self.current_provider = "ollama"
        self.providers = {
            "ollama": self.ollama_client
            # Add more providers here as needed
            # "openai": OpenAIClient(),
        }

    def generate_response(self, 
                         prompt: str, 
                         system_prompt: Optional[str] = None,
                         provider: Optional[str] = None) -> str:
        """
        Generate a response using the specified or current provider
        """
        active_provider = provider or self.current_provider
        if active_provider not in self.providers:
            raise ValueError(f"Provider {active_provider} not supported")
            
        return self.providers[active_provider].generate_response(
            prompt=prompt,
            system_prompt=system_prompt
        )

    def switch_provider(self, provider: str) -> None:
        """
        Switch between different LLM providers
        """
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not supported")
        self.current_provider = provider

    def set_model(self, model_name: str, provider: Optional[str] = None) -> None:
        """
        Set the model for a specific provider
        """
        active_provider = provider or self.current_provider
        if active_provider not in self.providers:
            raise ValueError(f"Provider {active_provider} not supported")
            
        self.providers[active_provider].set_model(model_name)

    def clear_history(self, provider: Optional[str] = None) -> None:
        """
        Clear conversation history for a specific provider
        """
        active_provider = provider or self.current_provider
        if active_provider not in self.providers:
            raise ValueError(f"Provider {active_provider} not supported")
            
        self.providers[active_provider].clear_history()

    def get_history(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """
        Get conversation history from a specific provider
        """
        active_provider = provider or self.current_provider
        if active_provider not in self.providers:
            raise ValueError(f"Provider {active_provider} not supported")
            
        return self.providers[active_provider].get_history()
