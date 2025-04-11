from typing import Optional, Dict, Any, List
import os
import logging
from dotenv import load_dotenv
import ollama
import anthropic
from openai import OpenAI

# Setup logging
from ..utils.logging import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

load_dotenv()

class LLMManager:
    def __init__(self):
        # Initialize clients
        self.ollama_client = ollama.Client(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        self.anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Default configurations - Read from environment or use fallbacks
        self.current_provider = "ollama" # Default provider
        self.current_models = {
            "ollama": os.getenv("DEFAULT_OLLAMA_MODEL", "qwen2.5:14b"),
            "anthropic": os.getenv("DEFAULT_ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
            "openai": os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4o-mini")
        }

        # Conversation histories for each provider
        self.conversation_histories: Dict[str, List[Dict[str, str]]] = {
            "ollama": [],
            "anthropic": [],
            "openai": []
        }

    def _calculate_temperature(self, personality_strength: Optional[float] = 0.5) -> float:
        """Calculates temperature based on personality strength. Range: [0.1, 1.0]"""
        strength = personality_strength if personality_strength is not None else 0.5
        # Inverse relationship: high strength -> low temp; low strength -> high temp
        # Clamp strength between 0 and 1
        clamped_strength = max(0.0, min(1.0, strength))
        temperature = 1.0 - clamped_strength * 0.9
        return max(0.1, temperature) # Ensure temperature is at least 0.1

    def generate_with_ollama(self, prompt: str, system_prompt: Optional[str] = None, personality_strength: Optional[float] = 0.5) -> str:
        """Generate response using Ollama"""
        messages = self.conversation_histories["ollama"].copy()
        temperature = self._calculate_temperature(personality_strength)
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.ollama_client.chat(
                model=self.current_models["ollama"],
                messages=messages,
                options={"temperature": temperature} # Pass temperature
            )

            assistant_message = response['message']['content']
            self.conversation_histories["ollama"].append({"role": "user", "content": prompt})
            self.conversation_histories["ollama"].append({"role": "assistant", "content": assistant_message})

            return assistant_message
        except Exception as e:
            logger.error(f"Error generating response with Ollama: {e}", exc_info=True)
            return ""

    def generate_with_anthropic(self, prompt: str, system_prompt: Optional[str] = None, personality_strength: Optional[float] = 0.5) -> str:
        """Generate response using Anthropic"""
        try:
            messages = []
            temperature = self._calculate_temperature(personality_strength)
            if system_prompt:
                # Anthropic API expects system prompt as a separate parameter
                pass # System prompt is handled below

            for msg in self.conversation_histories["anthropic"]:
                messages.append(msg)

            messages.append({"role": "user", "content": prompt})

            response = self.anthropic_client.messages.create(
                model=self.current_models["anthropic"],
                messages=messages,
                system=system_prompt, # Pass system prompt here
                temperature=temperature, # Pass temperature
                max_tokens=1024 # Define max tokens or adjust as needed
            )

            assistant_message = response.content[0].text
            
            self.conversation_histories["anthropic"].append({"role": "user", "content": prompt})
            self.conversation_histories["anthropic"].append({"role": "assistant", "content": assistant_message})
            return assistant_message
        except Exception as e:
            logger.error(f"Error generating response with Anthropic: {e}", exc_info=True)
            return ""

    def generate_with_openai(self, prompt: str, system_prompt: Optional[str] = None, personality_strength: Optional[float] = 0.5) -> str:
        """Generate response using OpenAI"""
        try:
            messages = []
            temperature = self._calculate_temperature(personality_strength)
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            for msg in self.conversation_histories["openai"]:
                messages.append(msg)
            
            messages.append({"role": "user", "content": prompt})
            
            response = self.openai_client.chat.completions.create(
                model=self.current_models["openai"],
                messages=messages,
                temperature=temperature # Pass temperature
            )

            assistant_message = response.choices[0].message.content
            self.conversation_histories["openai"].append({"role": "user", "content": prompt})
            self.conversation_histories["openai"].append({"role": "assistant", "content": assistant_message})

            return assistant_message
        except Exception as e:
            logger.error(f"Error generating response with OpenAI: {e}", exc_info=True)
            return ""

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None, personality_strength: Optional[float] = 0.5) -> str:
        """Generate response using current provider"""
        if self.current_provider == "ollama":
            return self.generate_with_ollama(prompt, system_prompt, personality_strength)
        elif self.current_provider == "anthropic":
            return self.generate_with_anthropic(prompt, system_prompt, personality_strength)
        elif self.current_provider == "openai":
            return self.generate_with_openai(prompt, system_prompt, personality_strength)
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
