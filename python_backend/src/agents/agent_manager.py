import os
import json
import yaml
import datetime
import logging
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any, Union

# Setup logging
from ..utils.logging import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

from ..llm.llm_manager import LLMManager
from ..memory.weaviate_client import WeaviateClient
from ..memory.mock_weaviate_client import MockWeaviateClient
from .agent import Agent
from .subconscious_agent import SubconsciousAgent

# Constants
MAX_LLM_RETRIES = 3
RETRY_DELAY_SECONDS = 1

class AgentManager:
    """
    Manages the lifecycle and coordination of multiple agents in the simulation.
    Handles agent turn processing including memory retrieval, response generation,
    XML parsing, error handling, and memory creation.
    """
    
    def __init__(self, llm_manager: LLMManager, weaviate_client: Optional[Union[WeaviateClient, MockWeaviateClient]] = None):
        """
        Initialize the AgentManager.
        
        Args:
            llm_manager: The LLM manager for agent responses
            weaviate_client: The Weaviate client for memory storage (or None to use mock)
        """
        self.llm_manager = llm_manager
        
        # If no Weaviate client is provided, use the mock implementation
        if weaviate_client is None:
            logger.warning("No Weaviate client provided, using mock implementation")
            self.weaviate_client = MockWeaviateClient("mock://localhost")
        else:
            self.weaviate_client = weaviate_client

        self.agents: Dict[str, Agent] = {}
        self.subconscious_agents: Dict[str, SubconsciousAgent] = {}
        self.config = self._load_config()
        self.tools = self._load_tools()
        self.memory_categories = self._load_memory_categories()
        self.environment_state = { # Basic environment state, can be updated externally
            "nearby_agents": {}, # {agent_name: [nearby_agent1, ...]}
            "nearby_objects": {} # {agent_name: [nearby_object1, ...]}
        }

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.yaml."""
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found at {config_path}")
            return {"paths": {}} # Return default empty paths
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file {config_path}: {e}")
            return {"paths": {}} # Return default empty paths

    def _get_config_path(self, key: str, default: str) -> str:
        """Helper to get a path from config, relative to src dir."""
        base_path = os.path.dirname(os.path.dirname(__file__)) # src directory
        relative_path = self.config.get("paths", {}).get(key, default)
        return os.path.join(base_path, relative_path)

    def _load_tools(self) -> Dict[str, Any]:
        """Load tools configuration from JSON file using path from config."""
        tools_path = self._get_config_path("tools_config", "config/tools.json")
        try:
            with open(tools_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Tools configuration file not found at {tools_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {tools_path}: {e}")
            return {}

    def _load_memory_categories(self) -> Dict[str, Any]:
        """Load memory categories configuration from JSON file using path from config."""
        categories_path = self._get_config_path("memory_categories_config", "config/memory_categories.json")
        try:
            with open(categories_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Memory categories file not found at {categories_path}")
            return {"categories": [], "importance_scale": {}} # Provide default structure
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {categories_path}: {e}")
            return {"categories": [], "importance_scale": {}} # Provide default structure

    def _load_agent_prompt(self, agent_name: str) -> str:
        """Load an agent's system prompt from its profile file or template."""
        profiles_dir = self._get_config_path("agent_profiles", "agents/profiles/")
        prompt_path = os.path.join(profiles_dir, f"{agent_name}.txt")
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"Profile file not found for agent '{agent_name}' at {prompt_path}. Using template.")
            # If specific agent prompt doesn't exist, use template
            templates_dir = self._get_config_path("prompt_templates", "agents/templates/")
            template_path = os.path.join(templates_dir, "agent_system_prompt_template.txt")
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    template = f.read()
                # Replace placeholder with agent name (personality etc. are replaced later)
                return template.replace("{{AGENT_NAME}}", agent_name)
            except FileNotFoundError:
                logger.error(f"Default agent prompt template not found at {template_path}")
                return f"You are {agent_name}. Default prompt template missing." # Basic fallback

    def create_agent(self, agent_name: str, personality: str, available_tools: List[str],
                     location: str = "starting_area", personality_strength: float = 0.5) -> Agent: # Add personality_strength
        """
        Create a new agent with its subconscious.

        Args:
            agent_name: Name of the agent
            personality: Description of agent's personality
            available_tools: List of tool names this agent can use
            location: Starting location of the agent
            personality_strength: Initial personality strength (0.0 to 1.0)

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
            system_prompt=system_prompt, # Base prompt, will be updated per turn
            available_tools=agent_tools,
            location=location,
            personality_strength=personality_strength
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
        logger.info(f"Created agent '{agent_name}' with subconscious.")
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
            logger.error(f"Attempted to process turn for non-existent agent: {agent_name}")
            raise ValueError(f"Agent {agent_name} does not exist")

        logger.debug(f"Processing turn for agent: {agent_name}")
        agent = self.agents[agent_name]
        subconscious = self.subconscious_agents[agent_name]

        # Update current datetime in agent's system prompt
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        updated_prompt = agent.system_prompt.replace("{{CURRENT_DATETIME}}", current_datetime)
        
        # Get relevant memories from subconscious
        logger.debug(f"Retrieving memories for {agent_name}...")
        memories = subconscious.retrieve_relevant_memories(agent.conversation_history, agent.location)
        memory_context = self._format_memories_for_prompt(memories)
        updated_prompt = updated_prompt.replace("{{MEMORY_CONTEXT}}", memory_context)
        logger.debug(f"Memory context for {agent_name}: {memory_context[:100]}...") # Log truncated context

        # Update nearby information (in a real implementation, this would come from the environment)
        # For now, we'll use placeholders
        # Update nearby information (replace placeholders with actual data if available)
        nearby_agents_str = ", ".join(self.environment_state.get("nearby_agents", {}).get(agent_name, [])) or "None"
        nearby_objects_str = ", ".join(self.environment_state.get("nearby_objects", {}).get(agent_name, [])) or "None"
        updated_prompt = updated_prompt.replace("{{NEARBY_AGENTS}}", nearby_agents_str)
        updated_prompt = updated_prompt.replace("{{NEARBY_OBJECTS}}", nearby_objects_str)

        # Add input message to conversation history *before* generating response
        if input_message:
            logger.debug(f"Adding input context to {agent_name}'s history: {input_message[:100]}...")
            agent.add_to_conversation("user", input_message) # Treat all input context as 'user' for the LLM

        # --- Generate Agent Response with Retry Logic ---
        raw_response = None
        processed_response = None
        last_error = None

        for attempt in range(MAX_LLM_RETRIES):
            logger.debug(f"Attempt {attempt + 1}/{MAX_LLM_RETRIES} to generate response for {agent_name}...")
            try:
                # Generate raw response using the Agent's method
                raw_response = agent.generate_response(updated_prompt)
                logger.debug(f"Raw response attempt {attempt + 1} from {agent_name}: {raw_response[:150]}...")

                # Process the response to extract reflection and tool use
                processed_response = self._process_agent_response(raw_response)

                # Basic validation: Check if essential parts exist
                if not processed_response.get("reflection") and not processed_response.get("tool_use"):
                     raise ValueError("Response missing both reflection and tool_use sections.")
                if not processed_response.get("tool_use", {}).get("name"):
                     logger.warning(f"Agent {agent_name} did not specify a tool. Assuming 'do_nothing'.")
                     # Optionally force a 'do_nothing' tool if none provided
                     # processed_response["tool_use"] = {"name": "do_nothing", "parameters": {}}

                # If successful, break the loop
                logger.info(f"Successfully generated and parsed response for {agent_name} on attempt {attempt + 1}.")
                last_error = None # Reset error on success
                break

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed for {agent_name}: {e}. Raw response: {raw_response}")
                if attempt < MAX_LLM_RETRIES - 1:
                    logger.info(f"Retrying after {RETRY_DELAY_SECONDS} seconds...")
                    time.sleep(RETRY_DELAY_SECONDS)
                    # Add error context for the next attempt
                    input_message += f"\n\n[SYSTEM_ERROR: Previous attempt failed: {e}. Please ensure response is valid XML with reflection and tool_use sections.]"
                    agent.add_to_conversation("user", f"[SYSTEM_ERROR: Previous attempt failed: {e}. Please ensure response is valid XML with reflection and tool_use sections.]") # Add to history too
                else:
                    logger.error(f"Agent {agent_name} failed to generate valid response after {MAX_LLM_RETRIES} attempts.")
                    # Handle final failure - return error state or default action
                    processed_response = {
                        "reflection": {"description": f"[ERROR: Failed to generate valid response after {MAX_LLM_RETRIES} attempts. Last error: {last_error}]"},
                        "tool_use": {"name": "do_nothing", "parameters": {}}, # Default to do_nothing on failure
                        "error": str(last_error),
                        "retrieved_memories": memories # Still include memories for context
                    }

        # Add retrieved memories to the final processed response
        processed_response["retrieved_memories"] = memories

        # --- Create Memory Post-Turn ---
        # Use the *successful* raw_response if available, otherwise use the error message
        final_assistant_content = raw_response if last_error is None else processed_response["reflection"]["description"]
        # Ensure the assistant message is added to history *after* successful generation/parsing or error handling
        if final_assistant_content:
             agent.add_to_conversation("assistant", final_assistant_content)

        # Create memory based on the *input* and the *final response/action*
        # Use the last two entries (user input + assistant response/error) for memory creation context
        memory_context_list = agent.conversation_history[-2:] if len(agent.conversation_history) >= 2 else agent.conversation_history
        if memory_context_list: # Only create memory if there was some interaction
            logger.debug(f"Creating memory for {agent_name} based on last interaction.")
            try:
                # Run memory creation in a separate thread to avoid blocking the main loop further
                # Note: This assumes Weaviate client is thread-safe or uses connection pooling.
                # If using mock client, this is fine.
                # For simplicity here, we'll run it synchronously, but consider threading for performance.
                created_memory = await asyncio.to_thread(
                     subconscious.create_memory_from_conversation,
                     memory_context_list,
                     agent.location
                )
                if created_memory:
                    logger.debug(f"Memory created successfully for {agent_name}. ID: {created_memory.get('id')}")
                else:
                    logger.warning(f"Memory creation returned empty for {agent_name}.")
            except Exception as e:
                logger.error(f"Failed to create memory for {agent_name}: {e}", exc_info=True)

        logger.info(f"Finished processing turn for agent: {agent_name}")
        return processed_response

    def _format_memories_for_prompt(self, memories: List[Dict[str, Any]]) -> str:
        """Format retrieved memories for inclusion in the agent's prompt."""
        if not memories:
            return "No relevant memories available."

        logger.debug(f"Formatting {len(memories)} memories for prompt.")
        memory_text = "Relevant memories:\n\n"
        for i, memory in enumerate(memories, 1):
            memory_text += f"Memory {i}:\n"
            memory_text += f"- Summary: {memory.get('summary', 'No summary')}\n"
            memory_text += f"- Category: {memory.get('category', 'Uncategorized')}\n"
            memory_text += f"- Time: {memory.get('timestamp', 'Unknown time')}\n"
            memory_text += f"- Critical Information: {memory.get('critical_information', 'None')}\n"
            memory_text += f"- Importance: {memory.get('importance', 5)}/10\n\n"
        return memory_text.strip()

    def _process_agent_response(self, xml_response: str) -> Dict[str, Any]:
        """
        Process the XML response from an agent using ElementTree.

        Args:
            xml_response: XML-formatted response string from the agent.

        Returns:
            Dictionary with parsed reflection and tool use information.
            Returns empty dicts if parsing fails or sections are missing.
        """
        result = {
            "reflection": {},
            "tool_use": {}
        }
        try:
            # Clean potential markdown code blocks if present
            cleaned_xml = xml_response.strip()
            if cleaned_xml.startswith("```xml"):
                cleaned_xml = cleaned_xml[len("```xml"):].strip()
            if cleaned_xml.endswith("```"):
                cleaned_xml = cleaned_xml[:-len("```")].strip()

            # Ensure there's a single root element for parsing robustness
            if not cleaned_xml.startswith("<response>"): # Assuming a root element might be missing
                 cleaned_xml = f"<response>{cleaned_xml}</response>"

            root = ET.fromstring(cleaned_xml)

            # Extract reflection components
            reflection_elements = ["current_task", "description", "next_steps", "goal", "other_info"]
            for elem_name in reflection_elements:
                element = root.find(elem_name)
                if element is not None and element.text:
                    result["reflection"][elem_name] = element.text.strip()
                else:
                     result["reflection"][elem_name] = "" # Ensure key exists even if empty

            # Extract tool use
            tool_use_element = root.find("tool_use")
            if tool_use_element is not None:
                tool_name_element = tool_use_element.find("tool_name")
                if tool_name_element is not None and tool_name_element.text:
                    result["tool_use"]["name"] = tool_name_element.text.strip()
                else:
                     logger.warning(f"Agent response XML missing <tool_name> within <tool_use>.")
                     result["tool_use"]["name"] = None # Explicitly set to None if missing

                parameters = {}
                for param_element in tool_use_element.findall("parameter"):
                    name = param_element.get("name")
                    value = param_element.text.strip() if param_element.text else ""
                    if name:
                        parameters[name] = value
                result["tool_use"]["parameters"] = parameters
            else:
                 logger.warning(f"Agent response XML missing <tool_use> section.")
                 # Set default tool_use if missing
                 result["tool_use"] = {"name": "do_nothing", "parameters": {}}


        except ET.ParseError as e:
            logger.error(f"XML parsing error in agent response: {e}\nInvalid XML: {xml_response[:500]}...", exc_info=True)
            # Raise the error so the retry logic in process_agent_turn can catch it
            raise ValueError(f"Invalid XML format: {e}")
        except Exception as e:
            logger.error(f"Unexpected error processing agent response XML: {e}\nXML: {xml_response[:500]}...", exc_info=True)
            # Raise the error
            raise ValueError(f"Unexpected error parsing XML: {e}")

        return result

    def get_all_agents(self) -> List[str]:
        """Get a list of all agent names."""
        return list(self.agents.keys())
    
    def delete_agent(self, agent_name: str) -> bool:
        """Delete an agent and its subconscious."""
        if agent_name in self.agents:
            del self.agents[agent_name]
            del self.subconscious_agents[agent_name]
            logger.info(f"Deleted agent: {agent_name}")
            return True
        logger.warning(f"Attempted to delete non-existent agent: {agent_name}")
        return False
