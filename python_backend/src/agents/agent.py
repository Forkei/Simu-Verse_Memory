from typing import Dict, List, Any, Optional
from ..llm.llm_manager import LLMManager

class Agent:
    """
    Base class for all agents in the simulation.
    """
    
    def __init__(self, name: str, llm_manager: LLMManager, system_prompt: str, 
                 available_tools: Dict[str, Any], location: str = "starting_area"):
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
    
    def generate_response(self, updated_system_prompt: str) -> str:
        """
        Generate a response from the agent using the LLM.
        
        Args:
            updated_system_prompt: System prompt with current context
            
        Returns:
            The agent's response in XML format
        """
        # Construct the prompt from conversation history
        prompt = self._construct_prompt()
        
        # Generate response using the LLM
        response = self.llm_manager.generate_response(prompt, updated_system_prompt)
        
        # Add the response to conversation history
        self.add_to_conversation("assistant", response)
        
        return response
    
    def _construct_prompt(self) -> str:
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
