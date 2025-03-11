import os
import sys
from dotenv import load_dotenv
from python_backend.src.llm.llm_manager import LLMManager
from python_backend.src.memory.weaviate_client import WeaviateClient
from python_backend.src.agents.agent_manager import AgentManager

def main():
    """
    Main entry point for the Simu-Verse agent system.
    """
    # Load environment variables
    load_dotenv()
    
    # Initialize LLM manager
    llm_manager = LLMManager()
    
    # Initialize Weaviate client
    weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    weaviate_client = WeaviateClient(weaviate_url)
    
    # Initialize agent manager
    agent_manager = AgentManager(llm_manager, weaviate_client)
    
    # Create an agent
    alice = agent_manager.create_agent(
        agent_name="alice",
        personality="You are a friendly and helpful AI researcher who specializes in knowledge management and information retrieval.",
        available_tools=["movement", "scan", "talk", "think", "interact"],
        location="research_lab"
    )
    
    # Process a turn for the agent
    response = agent_manager.process_agent_turn("alice", "Hello Alice, can you help me find some research papers?")
    
    # Print the response
    print("\nAgent Response:")
    print("---------------")
    print(response)
    
    # Process another turn
    response = agent_manager.process_agent_turn("alice", "I'm specifically looking for papers on memory systems.")
    
    # Print the response
    print("\nAgent Response:")
    print("---------------")
    print(response)

if __name__ == "__main__":
    main()
