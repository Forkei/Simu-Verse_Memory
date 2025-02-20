from typing import Optional, Dict, Any, List
import os
from dotenv import load_dotenv
import ollama
import anthropic
from openai import OpenAI

load_dotenv()

class LLMManager:
    def __init__(self):
        # Initialize clients
        self.ollama_client = ollama.Client(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        self.anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Default configurations
        self.current_provider = "ollama"
        self.current_models = {
            "ollama": "llama2",
            "anthropic": "claude-3-opus-20240229",
            "openai": "gpt-4-turbo-preview"
        }
        
        # Conversation histories for each provider
        self.conversation_histories: Dict[str, List[Dict[str, str]]] = {
            "ollama": [],
            "anthropic": [],
            "openai": []
        }

    def generate_with_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response using Ollama"""
        messages = self.conversation_histories["ollama"].copy()
        
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.ollama_client.chat(
                model=self.current_models["ollama"],
                messages=messages
            )
            
            assistant_message = response['message']['content']
            self.conversation_histories["ollama"].append({"role": "user", "content": prompt})
            self.conversation_histories["ollama"].append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
        except Exception as e:
            print(f"Error with Ollama: {e}")
            return ""

    def generate_with_anthropic(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response using Anthropic"""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            for msg in self.conversation_histories["anthropic"]:
                messages.append(msg)
            
            messages.append({"role": "user", "content": prompt})
            
            response = self.anthropic_client.messages.create(
                model=self.current_models["anthropic"],
                messages=messages
            )
            
            assistant_message = response.content[0].text
            self.conversation_histories["anthropic"].append({"role": "user", "content": prompt})
            self.conversation_histories["anthropic"].append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
        except Exception as e:
            print(f"Error with Anthropic: {e}")
            return ""

    def generate_with_openai(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response using OpenAI"""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            for msg in self.conversation_histories["openai"]:
                messages.append(msg)
            
            messages.append({"role": "user", "content": prompt})
            
            response = self.openai_client.chat.completions.create(
                model=self.current_models["openai"],
                messages=messages
            )
            
            assistant_message = response.choices[0].message.content
            self.conversation_histories["openai"].append({"role": "user", "content": prompt})
            self.conversation_histories["openai"].append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
        except Exception as e:
            print(f"Error with OpenAI: {e}")
            return ""

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response using current provider"""
        if self.current_provider == "ollama":
            return self.generate_with_ollama(prompt, system_prompt)
        elif self.current_provider == "anthropic":
            return self.generate_with_anthropic(prompt, system_prompt)
        elif self.current_provider == "openai":
            return self.generate_with_openai(prompt, system_prompt)
        else:
            raise ValueError(f"Provider {self.current_provider} not supported")

    def switch_provider(self, provider: str) -> None:
        """Switch between different LLM providers"""
        if provider not in ["ollama", "anthropic", "openai"]:
            raise ValueError(f"Provider {provider} not supported")
        self.current_provider = provider

    def set_model(self, model_name: str, provider: Optional[str] = None) -> None:
        """Set the model for a specific provider"""
        active_provider = provider or self.current_provider
        if active_provider not in self.current_models:
            raise ValueError(f"Provider {active_provider} not supported")
        self.current_models[active_provider] = model_name

    def clear_history(self, provider: Optional[str] = None) -> None:
        """Clear conversation history for a specific provider"""
        active_provider = provider or self.current_provider
        if active_provider not in self.conversation_histories:
            raise ValueError(f"Provider {active_provider} not supported")
        self.conversation_histories[active_provider] = []

    def get_history(self, provider: Optional[str] = None) -> List[Dict[str, str]]:
        """Get conversation history from a specific provider"""
        active_provider = provider or self.current_provider
        if active_provider not in self.conversation_histories:
            raise ValueError(f"Provider {active_provider} not supported")
        return self.conversation_histories[active_provider]
