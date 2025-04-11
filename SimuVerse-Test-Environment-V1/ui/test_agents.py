import streamlit as st
import os
from time import sleep
from dotenv import load_dotenv
import sys
import json # Add this import

# Adjust path to import from python_backend
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'python_backend', 'src'))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from llm.llm_manager import LLMManager
from memory.weaviate_client import WeaviateClient # Or MockWeaviateClient
from agents.agent_manager import AgentManager as BackendAgentManager

def generate_markdown(conversation_log):
    """Helper function to generate markdown for the conversation log."""
    md = ""
    for speaker, msg in conversation_log:
        md += f"**{speaker}:** {msg}\n\n"
    return md

# Load API key from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
claude_api_key = os.getenv("CLAUDE_API_KEY")
if not openai_api_key or not claude_api_key:
    # st.error("OpenAI or Claude API key not found in environment variables.")
    logger.critical("OpenAI or Claude API key not found in environment variables.")
    st.stop()

# Instantiate Managers (using backend components)
logger.info("Instantiating LLMManager...")
llm_manager = LLMManager() # Assumes API keys are loaded via dotenv in LLMManager itself
# Use MockWeaviateClient for this test script as Weaviate might not be running
from memory.mock_weaviate_client import MockWeaviateClient
logger.info("Using MockWeaviateClient.")
weaviate_client = MockWeaviateClient(url="mock://localhost") # Mock client doesn't need a real URL

logger.info("Instantiating BackendAgentManager...")
backend_agent_manager = BackendAgentManager(llm_manager=llm_manager, weaviate_client=weaviate_client)

# Load Tools (needed for agent creation)
tools_path = os.path.join(backend_path, "config", "tools.json")
try:
    with open(tools_path, 'r') as f:
        tools_config = json.load(f)
    all_tool_names = list(tools_config.keys())
    logger.info(f"Loaded tools: {all_tool_names}")
except Exception as e:
    # st.error(f"Failed to load tools from {tools_path}: {e}")
    logger.critical(f"Failed to load tools from {tools_path}: {e}", exc_info=True)
    st.stop()

st.title("Multi-Agent Conversation Test")
st.write("This app demonstrates the backend agent system.")

# Create two agents using the backend AgentManager
james_personality = (
    "You are James, a friendly conversation partner who is a 20 yr old male. "
    "Please respond naturally in a conversational tone and limit your reply to no more than 2 sentences."
)
james = backend_agent_manager.create_agent(
    agent_name="James",
    personality=james_personality,
    available_tools=all_tool_names,
    location="test_environment"
)
logger.info(f"Created agent: {jade.name}")
logger.info(f"Created agent: {james.name}")

jade_personality = (
    "You are Jade, an engaging conversation expert who is a 20 yr old female."
    "Respond in a concise, human-like manner using no more than 2 sentences."
)
jade = backend_agent_manager.create_agent(
    agent_name="Jade",
    personality=jade_personality,
    available_tools=all_tool_names,
    location="test_environment"
)

# Input for the initial message
initial_message = st.text_input(
    "Initial message:", 
    value="Have a conversation with your partner, begin by talking about your lives"
)

# Number of rounds (default is 3 rounds)
num_rounds = st.number_input("Number of rounds", min_value=1, max_value=10, value=3, step=1)

if st.button("Start Conversation") and initial_message:
    with st.expander("Backend Agent Conversation", expanded=True):
        conversation_log = []
        chat_placeholder = st.empty()
        
        # Use backend agent manager to process turns
        current_speaker = "James"
        message_to_process = initial_message
        
        for i in range(num_rounds * 2): # Each round has two turns
            target_agent_name = current_speaker

            # st.write(f"Processing turn for: {target_agent_name}")
            logger.info(f"Processing turn {i+1} for: {target_agent_name}")

            # Process turn using backend manager
            try:
                processed_response = backend_agent_manager.process_agent_turn(target_agent_name, message_to_process)
                logger.debug(f"Response received from {target_agent_name}")
            except Exception as turn_error:
                logger.error(f"Error processing turn for {target_agent_name}: {turn_error}", exc_info=True)
                st.error(f"Error processing turn for {target_agent_name}: {turn_error}")
                break # Stop the conversation on error

            # Extract display message (simplified for this test)
            reflection = processed_response.get("reflection", {})
            display_response = reflection.get("description", "...") # Or use other fields
            
            conversation_log.append((target_agent_name, display_response))
            chat_placeholder.markdown(generate_markdown(conversation_log))
            sleep(1) # Optional pause
            
            # Prepare for next turn
            message_to_process = display_response # Next agent responds to this
            current_speaker = "Jade" if current_speaker == "James" else "James"
            
        # Display final state/info if needed
        st.write("**Final Agent Info:**")
        st.write(f"James last response details: {james.last_processed_response}")
        st.write(f"Jade last response details: {jade.last_processed_response}")
