import os
import json
import datetime
from typing import Dict, List, Optional, Any
from ..llm.llm_manager import LLMManager
from ..memory.weaviate_client import WeaviateClient
from .agent import Agent
from .subconscious_agent import SubconsciousAgent

class AgentManager:
    """
    Manages the lifecycle and coordination of multiple agents in the simulation.
    """
    
    def __init__(self, llm_manager: LLMManager, weaviate_client: WeaviateClient):
        """
        Initialize the AgentManager.
        
        Args:
            llm_manager: The LLM manager for agent responses
            weaviate_client: The Weaviate client for memory storage
        """
        self.llm_manager = llm_manager
        self.weaviate_client = weaviate_client
        self.agents: Dict[str, Agent] = {}
        self.subconscious_agents: Dict[str, SubconsciousAgent] = {}
        self.tools = self._load_tools()
        self.memory_categories = self._load_memory_categories()
        
    def _load_tools(self) -> Dict[str, Any]:
        """Load tools configuration from JSON file."""
        tools_path = os.path.join(os.path.dirname(__file__), "..", "config", "tools.json")
        with open(tools_path, 'r') as f:
            return json.load(f)
    
    def _load_memory_categories(self) -> Dict[str, Any]:
        """Load memory categories configuration from JSON file."""
        categories_path = os.path.join(os.path.dirname(__file__), "..", "config", "memory_categories.json")
        with open(categories_path, 'r') as f:
            return json.load(f)
    
    def _load_agent_prompt(self, agent_name: str) -> str:
        """Load an agent's system prompt from its text file."""
        prompt_path = os.path.join(os.path.dirname(__file__), "profiles", f"{agent_name}.txt")
        try:
            with open(prompt_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            # If specific agent prompt doesn't exist, use template
            template_path = os.path.join(os.path.dirname(__file__), "templates", "agent_system_prompt_template.txt")
            with open(template_path, 'r') as f:
                template = f.read()
            
            # Replace placeholder with agent name
            return template.replace("{{AGENT_NAME}}", agent_name)
    
    def create_agent(self, agent_name: str, personality: str, available_tools: List[str], location: str = "starting_area") -> Agent:
        """
        Create a new agent with its subconscious.
        
        Args:
            agent_name: Name of the agent
            personality: Description of agent's personality
            available_tools: List of tool names this agent can use
            location: Starting location of the agent
            
        Returns:
            The created agent
        """
        # Create agent's memory collection in Weaviate if it doesn't exist
        collection_name = f"Memories_{agent_name}"
        self.weaviate_client.create_collection_if_not_exists(collection_name)
        
        # Load system prompt and inject agent-specific information
        system_prompt = self._load_agent_prompt(agent_name)
        system_prompt = system_prompt.replace("{{AGENT_NAME}}", agent_name)
        system_prompt = system_prompt.replace("{{AGENT_PERSONALITY}}", personality)
        system_prompt = system_prompt.replace("{{CURRENT_LOCATION}}", location)
        
        # Filter tools to only those available to this agent
        agent_tools = {name: self.tools[name] for name in available_tools if name in self.tools}
        tools_info = self._format_tools_for_prompt(agent_tools)
        system_prompt = system_prompt.replace("{{TOOLS_INFO}}", tools_info)
        
        # Create the agent
        agent = Agent(
            name=agent_name,
            llm_manager=self.llm_manager,
            system_prompt=system_prompt,
            available_tools=agent_tools,
            location=location
        )
        
        # Create the subconscious agent
        subconscious = SubconsciousAgent(
            agent_name=agent_name,
            llm_manager=self.llm_manager,
            weaviate_client=self.weaviate_client,
            memory_categories=self.memory_categories
        )
        
        # Store both agents
        self.agents[agent_name] = agent
        self.subconscious_agents[agent_name] = subconscious
        
        return agent
    
    def _format_tools_for_prompt(self, tools: Dict[str, Any]) -> str:
        """Format tools information for inclusion in system prompt."""
        tools_text = ""
        for name, info in tools.items():
            tools_text += f"Tool: {name}\n"
            tools_text += f"Description: {info['description']}\n"
            tools_text += "Parameters:\n"
            
            for param_name, param_info in info['parameters'].items():
                required = "Required" if param_info.get('required', False) else "Optional"
                default = f", Default: {param_info.get('default')}" if 'default' in param_info else ""
                enum_values = f", Values: {', '.join(param_info['enum'])}" if 'enum' in param_info else ""
                
                tools_text += f"  - {param_name}: {param_info['description']} ({required}{default}{enum_values})\n"
            
            tools_text += "\n"
        
        return tools_text
    
    def process_agent_turn(self, agent_name: str, input_message: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a turn for an agent, including memory retrieval and response generation.
        
        Args:
            agent_name: Name of the agent to process
            input_message: Optional input message to the agent
            
        Returns:
            Dictionary containing the agent's response and actions
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent {agent_name} does not exist")
        
        agent = self.agents[agent_name]
        subconscious = self.subconscious_agents[agent_name]
        
        # Update current datetime in agent's system prompt
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        updated_prompt = agent.system_prompt.replace("{{CURRENT_DATETIME}}", current_datetime)
        
        # Get relevant memories from subconscious
        memories = subconscious.retrieve_relevant_memories(agent.conversation_history, agent.location)
        memory_context = self._format_memories_for_prompt(memories)
        updated_prompt = updated_prompt.replace("{{MEMORY_CONTEXT}}", memory_context)
        
        # Update nearby information (in a real implementation, this would come from the environment)
        # For now, we'll use placeholders
        updated_prompt = updated_prompt.replace("{{NEARBY_AGENTS}}", "None")
        updated_prompt = updated_prompt.replace("{{NEARBY_OBJECTS}}", "None")
        
        # Generate agent response
        if input_message:
            agent.add_to_conversation("user", input_message)
        
        response = agent.generate_response(updated_prompt)
        
        # Process the response to extract reflection and tool use
        processed_response = self._process_agent_response(response)
        
        # Create a new memory from this interaction
        if input_message or len(agent.conversation_history) > 0:
            subconscious.create_memory_from_conversation(
                agent.conversation_history[-10:] if len(agent.conversation_history) >= 10 else agent.conversation_history,
                agent.location
            )
        
        return processed_response
    
    def _format_memories_for_prompt(self, memories: List[Dict[str, Any]]) -> str:
        """Format retrieved memories for inclusion in the agent's prompt."""
        if not memories:
            return "No relevant memories available."
        
        memory_text = "Relevant memories:\n\n"
        for i, memory in enumerate(memories, 1):
            memory_text += f"Memory {i}:\n"
            memory_text += f"- Summary: {memory.get('summary', 'No summary')}\n"
            memory_text += f"- Category: {memory.get('category', 'Uncategorized')}\n"
            memory_text += f"- Time: {memory.get('timestamp', 'Unknown time')}\n"
            memory_text += f"- Critical Information: {memory.get('critical_information', 'None')}\n"
            memory_text += f"- Importance: {memory.get('importance', 5)}/10\n\n"
        
        return memory_text
    
    def _process_agent_response(self, response: str) -> Dict[str, Any]:
        """
        Process the XML response from an agent to extract reflection and tool use.
        
        Args:
            response: XML-formatted response from the agent
            
        Returns:
            Dictionary with parsed reflection and tool use information
        """
        # This is a simplified parser - in a real implementation, use a proper XML parser
        result = {
            "reflection": {},
            "tool_use": {}
        }
        
        # Extract reflection components
        for component in ["current_task", "description", "next_steps", "goal", "other_info"]:
            start_tag = f"<{component}>"
            end_tag = f"</{component}>"
            if start_tag in response and end_tag in response:
                start_idx = response.find(start_tag) + len(start_tag)
                end_idx = response.find(end_tag)
                result["reflection"][component] = response[start_idx:end_idx].strip()
        
        # Extract tool use
        if "<tool_use>" in response and "</tool_use>" in response:
            tool_section = response.split("<tool_use>")[1].split("</tool_use>")[0].strip()
            
            # Extract tool name
            if "<tool_name>" in tool_section and "</tool_name>" in tool_section:
                start_idx = tool_section.find("<tool_name>") + len("<tool_name>")
                end_idx = tool_section.find("</tool_name>")
                result["tool_use"]["name"] = tool_section[start_idx:end_idx].strip()
            
            # Extract parameters
            result["tool_use"]["parameters"] = {}
            param_sections = tool_section.split("<parameter ")
            for section in param_sections[1:]:  # Skip the first empty split
                if ">" in section and "</parameter>" in section:
                    # Extract parameter name
                    name_start = section.find('name="') + 6
                    name_end = section.find('"', name_start)
                    param_name = section[name_start:name_end]
                    
                    # Extract parameter value
                    value_start = section.find(">") + 1
                    value_end = section.find("</parameter>")
                    param_value = section[value_start:value_end].strip()
                    
                    result["tool_use"]["parameters"][param_name] = param_value
        
        return result
    
    def get_all_agents(self) -> List[str]:
        """Get a list of all agent names."""
        return list(self.agents.keys())
    
    def delete_agent(self, agent_name: str) -> bool:
        """Delete an agent and its subconscious."""
        if agent_name in self.agents:
            del self.agents[agent_name]
            del self.subconscious_agents[agent_name]
            return True
        return False
