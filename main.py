import os
import sys
import logging
from dotenv import load_dotenv

# Setup logging first
from python_backend.src.utils.logging import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

from python_backend.src.llm.llm_manager import LLMManager
from python_backend.src.memory.weaviate_client import WeaviateClient
from python_backend.src.memory.memory_utils import MemoryUtils
from python_backend.src.agents.agent_manager import AgentManager


def main():
    """
    Main entry point for the Simu-Verse agent system.
    """
    # Load environment variables
    load_dotenv()
    
    logger.info("Starting Simu-Verse Agent System...")
    try:
        # Initialize LLM manager
        llm_manager = LLMManager()
        logger.info("Initialized LLM manager")

        # Initialize Weaviate client
        weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

        # Try to connect to Weaviate
        weaviate_client = None # Initialize as None
        try:
            logger.info(f"Attempting to connect to Weaviate at {weaviate_url}")
            weaviate_client = WeaviateClient(weaviate_url, weaviate_api_key)
            # weaviate_client.set_llm_manager(llm_manager) # LLM Manager not needed directly by WeaviateClient anymore
            logger.info(f"Successfully connected to Weaviate at {weaviate_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}", exc_info=True)
            logger.warning("Running in memory-less mode. Memories will not be persisted.")
            # weaviate_client remains None

        # Initialize memory utilities (can work without Weaviate for some functions)
        memory_utils = MemoryUtils(llm_manager)
        logger.info("Initialized memory utilities")

        # Initialize agent manager
        agent_manager = AgentManager(llm_manager, weaviate_client) # Pass None if connection failed
        logger.info("Initialized agent manager")

        # Create an agent
        alice = agent_manager.create_agent(
            agent_name="alice",
            personality="You are a friendly and helpful AI researcher who specializes in knowledge management and information retrieval.",
            available_tools=["movement", "scan", "talk", "think", "interact"],
            location="research_lab"
        )
        # logging.info(f"Created agent: alice") # AgentManager logs this now

        # Interactive mode
        print("\nWelcome to Simu-Verse Agent System!")
        print("Type 'exit' to quit, 'help' for commands\n")

        current_agent_name = "alice" # Keep track of the active agent

        while True:
            user_input = input(f"\n{current_agent_name} > ")

            if user_input.lower() in ['exit', 'quit']:
                logger.info("Exiting Simu-Verse Agent System.")
                break

            if user_input.lower() == 'help':
                print("\nCommands:")
                print("  exit/quit       - Exit the program")
                print("  help            - Show this help message")
                print("  agents          - List all agents")
                print("  create <name>   - Create a new agent")
                print("  switch <name>   - Switch to interact with a different agent")
                print("  Any other input will be sent to the current agent")
                continue

            if user_input.lower() == 'agents':
                agents = agent_manager.get_all_agents()
                print("\nAvailable agents:")
                for agent_name in agents:
                    print(f"  - {agent_name}")
                continue

            if user_input.lower().startswith('create '):
                new_agent_name = user_input[7:].strip()
                if not new_agent_name:
                    print("Please specify an agent name.")
                    continue
                if new_agent_name in agent_manager.agents:
                    print(f"Agent '{new_agent_name}' already exists.")
                    continue

                # personality = input("Enter personality description (or leave blank to use profile): ")
                # For simplicity in CLI, we'll rely on profile files
                try:
                    agent_manager.create_agent(
                        agent_name=new_agent_name,
                        personality="", # Loaded from file
                        available_tools=["movement", "scan", "talk", "think", "interact", "do_nothing"],
                        location="command_line_env"
                    )
                    # logger.info(f"Created agent: {new_agent_name}") # Logged by AgentManager
                except Exception as create_error:
                    logger.error(f"Failed to create agent '{new_agent_name}': {create_error}", exc_info=True)
                    print(f"Error creating agent: {create_error}")
                continue

            if user_input.lower().startswith('switch '):
                target_agent = user_input[7:].strip()
                if target_agent in agent_manager.agents:
                    current_agent_name = target_agent
                    print(f"Switched to agent: {target_agent}")
                else:
                    print(f"Agent '{target_agent}' not found.")
                continue

            # Process the user input with the current agent
            try:
                logger.info(f"Processing input for {current_agent_name}: '{user_input[:50]}...'")
                response = agent_manager.process_agent_turn(current_agent_name, user_input)

                print(f"\n{current_agent_name} Response:")
                print("--------------------")

                # Print reflection
                if "reflection" in response:
                    for key, value in response["reflection"].items():
                        print(f"{key.replace('_', ' ').title()}: {value}")
                
                # Print tool use
                if "tool_use" in response and "name" in response["tool_use"]:
                    print("\nTool Used:")
                    print(f"  {response['tool_use']['name']}")
                    
                    if "parameters" in response["tool_use"]:
                        print("  Parameters:")
                        for param, value in response["tool_use"]["parameters"].items():
                            print(f"    {param}: {value}")
                else:
                    print("  (No tool used)")

            except Exception as e:
                logger.error(f"Error processing agent turn for {current_agent_name}: {e}", exc_info=True)
                print(f"Error: {e}")

    except Exception as e:
        logger.critical(f"Fatal error during initialization or main loop: {e}", exc_info=True)
        print(f"Fatal error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
