from typing import Dict, List, Any, Optional
from ..llm.llm_manager import LLMManager

class Agent:
    """
    Base class for all agents in the simulation.
    """
    
    def __init__(self, name: str, llm_manager: LLMManager, system_prompt: str,
                 available_tools: Dict[str, Any], location: str = "starting_area",
                 personality_strength: float = 0.5): # Add personality_strength
        """
        Initialize an agent.
        Args:
            name: The agent's name
            llm_manager: LLM manager for generating responses
            system_prompt: The agent's system prompt
            available_tools: Dictionary of tools available to this agent
            location: The agent's current location
        """
        self.name = name
        self.llm_manager = llm_manager
        self.system_prompt = system_prompt
        self.available_tools = available_tools
        self.location = location
        self.conversation_history: List[Dict[str, str]] = []
        self.thinking: bool = False # For UI state tracking
        self.last_processed_response: Optional[Dict[str, Any]] = None # Store result from AgentManager
        self.last_scan_result: Optional[str] = None # Store scan result for next turn's input
        self.personality_strength: float = personality_strength # Store personality strength

    def generate_response(self, turn_system_prompt: str) -> str:
        """
        Generate a response from the agent using the LLM.
        The AgentManager prepares the full context (history, memories, etc.)
        and passes it via the turn_system_prompt and the last user message.

        Args:
            turn_system_prompt: The complete system prompt for this specific turn,
                                including personality, location, memories, tools etc.

        Returns:
            The agent's raw XML response string.
        """
        # The prompt is now just the last user message (which might contain system info, memories etc.)
        # The system prompt contains the rest of the context.
        # History is managed by LLMManager based on the provider.
        last_user_message = ""
        if self.conversation_history and self.conversation_history[-1]["role"] == "user":
             last_user_message = self.conversation_history[-1]["content"]
        else:
             # This case should ideally be handled by AgentManager providing an initial prompt
             logger.warning(f"Agent {self.name} generating response without preceding user message in history.")
             last_user_message = "[SYSTEM: Start of interaction or previous context missing. Please respond based on your current state.]"


        # Generate response using the LLM, passing personality strength
        # The LLMManager will handle combining the system prompt, history, and the new user message.
        response = self.llm_manager.generate_response(
            prompt=last_user_message, # The actual new input/stimulus
            system_prompt=turn_system_prompt, # The full context prompt
            personality_strength=self.personality_strength
        )

        # Note: Adding the assistant response to history is now handled by AgentManager *after* successful parsing.

        return response

    # _construct_prompt is no longer needed here as context is built in AgentManager
    # def _construct_prompt(self) -> str:
    #     """
    #     Construct a prompt from the conversation history. (DEPRECATED)
    #
    #     Returns:
    #         The constructed prompt
    #     """
    #     if not self.conversation_history:
    #         return "Please respond to this initial interaction."
    #
    #     # Get the last few messages from conversation history
    #     recent_messages = self.conversation_history[-5:]  # Last 5 messages
    #
    #     # Construct the prompt
    #     prompt = "Based on our conversation so far, please respond to this interaction:\n\n"
    #
    #     for message in recent_messages:
    #         role = message["role"]
    #         content = message["content"]
    #
    #         if role == "user":
    #             prompt += f"User: {content}\n\n"
    #         else:
    #             prompt += f"You: {content}\n\n"
    #
    #     return prompt

    def add_to_conversation(self, role: str, content: str) -> None:
        """
        Construct a prompt from the conversation history.
        
        Returns:
            The constructed prompt
        """
        if not self.conversation_history:
            return "Please respond to this initial interaction."
        
        # Get the last few messages from conversation history
        recent_messages = self.conversation_history[-5:]  # Last 5 messages
        
        # Construct the prompt
        prompt = "Based on our conversation so far, please respond to this interaction:\n\n"
        
        for message in recent_messages:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                prompt += f"User: {content}\n\n"
            else:
                prompt += f"You: {content}\n\n"
        
        return prompt
    
    def add_to_conversation(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role: The role of the message sender ("user" or "assistant")
            content: The content of the message
        """
        self.conversation_history.append({
            "role": role,
            "content": content
        })
    
    def move_to(self, new_location: str) -> None:
        """
        Move the agent to a new location.
        
        Args:
            new_location: The new location
        """
        self.location = new_location
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool with the given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters for the tool
            
        Returns:
            Result of the tool execution
        """
        # In a real implementation, this would connect to the environment
        # For now, we'll just return a placeholder result
        if tool_name not in self.available_tools:
            return {"error": f"Tool {tool_name} not available to this agent"}
        
        # Validate parameters
        tool_config = self.available_tools[tool_name]
        for param_name, param_config in tool_config["parameters"].items():
            if param_config.get("required", False) and param_name not in parameters:
                return {"error": f"Required parameter {param_name} missing for tool {tool_name}"}
        
        # Return a placeholder result
        return {
            "tool": tool_name,
            "parameters": parameters,
            "result": f"Executed {tool_name} with parameters {parameters}"
        }
