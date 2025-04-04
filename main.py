import os
import sys
import logging
from dotenv import load_dotenv
from python_backend.src.llm.llm_manager import LLMManager
from python_backend.src.memory.weaviate_client import WeaviateClient
from python_backend.src.memory.memory_utils import MemoryUtils
from python_backend.src.agents.agent_manager import AgentManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('simu_verse.log')
    ]
)

def main():
    """
    Main entry point for the Simu-Verse agent system.
    """
    # Load environment variables
    load_dotenv()
    
    try:
        # Initialize LLM manager
        llm_manager = LLMManager()
        logging.info("Initialized LLM manager")
        
        # Initialize Weaviate client
        weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
        
        # Try to connect to Weaviate
        try:
            logging.info(f"Attempting to connect to Weaviate at {weaviate_url}")
            weaviate_client = WeaviateClient(weaviate_url, weaviate_api_key)
            weaviate_client.set_llm_manager(llm_manager)
            logging.info(f"Successfully connected to Weaviate at {weaviate_url}")
        except Exception as e:
            logging.error(f"Failed to connect to Weaviate: {e}")
            logging.warning("Running in memory-less mode. Memories will not be persisted.")
            weaviate_client = None
            
        # Initialize memory utilities
        memory_utils = MemoryUtils(llm_manager)
        
        # Initialize agent manager
        agent_manager = AgentManager(llm_manager, weaviate_client)
        logging.info("Initialized agent manager")
        
        # Create an agent
        alice = agent_manager.create_agent(
            agent_name="alice",
            personality="You are a friendly and helpful AI researcher who specializes in knowledge management and information retrieval.",
            available_tools=["movement", "scan", "talk", "think", "interact"],
            location="research_lab"
        )
        logging.info(f"Created agent: alice")
        
        # Interactive mode
        print("\nWelcome to Simu-Verse Agent System!")
        print("Type 'exit' to quit, 'help' for commands\n")
        
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ['exit', 'quit']:
                break
                
            if user_input.lower() == 'help':
                print("\nCommands:")
                print("  exit/quit - Exit the program")
                print("  help - Show this help message")
                print("  agents - List all agents")
                print("  create <name> - Create a new agent")
                print("  switch <name> - Switch to a different agent")
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
                    print("Please specify an agent name")
                    continue
                    
                personality = input("Enter personality description: ")
                agent_manager.create_agent(
                    agent_name=new_agent_name,
                    personality=personality,
                    available_tools=["movement", "scan", "talk", "think", "interact", "do_nothing"],
                    location="starting_area"
                )
                print(f"Created agent: {new_agent_name}")
                continue
                
            if user_input.lower().startswith('switch '):
                target_agent = user_input[7:].strip()
                if target_agent in agent_manager.agents:
                    alice = agent_manager.agents[target_agent]
                    print(f"Switched to agent: {target_agent}")
                else:
                    print(f"Agent {target_agent} not found")
                continue
            
            # Process the user input with the current agent
            try:
                response = agent_manager.process_agent_turn("alice", user_input)
                
                print("\nAgent Response:")
                print("---------------")
                
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
                
            except Exception as e:
                logging.error(f"Error processing agent turn: {e}")
                print(f"Error: {e}")
    
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        print(f"Fatal error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
