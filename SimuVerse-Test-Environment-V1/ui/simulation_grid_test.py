import streamlit as st
# import os
import math
import dash
from dash import html, dcc, Input, Output, State, callback_context, no_update
import dash.long_callback # Import long_callback
from dash.long_callback import DiskcacheLongCallbackManager # Import manager
import diskcache # Import diskcache
import dash_cytoscape as cyto
from dash import dash_table
from dash import ALL
import dash_bootstrap_components as dbc
from dotenv import load_dotenv

import sys
import os
import json
import asyncio
import datetime # Add this import
from typing import Optional # Add this import

# Adjust path to import from python_backend
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'python_backend', 'src'))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from llm.llm_manager import LLMManager
from memory.weaviate_client import WeaviateClient
# Alias the backend AgentManager to avoid name collision
from agents.agent_manager import AgentManager as BackendAgentManager
from agents.agent import Agent as BackendAgent # Import backend Agent for type hinting if needed

# Remove the old create_agent_with_llm import if no longer used directly
# from src.agent_manager import create_agent_with_llm # This file will be deleted

# -------------------------
# Load API Keys, Instantiate Managers, Load Tools, and Create Agents
# -------------------------

load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
claude_api_key = os.getenv("CLAUDE_API_KEY")
weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080") # Default if not set

if not openai_api_key or not claude_api_key:
    st.error("OpenAI or Claude API key not found in environment variables.")
    st.stop()
if not weaviate_url:
    st.error("WEAVIATE_URL not found in environment variables.")
    st.stop()

# Instantiate Managers
llm_manager = LLMManager() # Assumes API keys are loaded via dotenv in LLMManager itself
try:
    # Assuming Weaviate doesn't need an API key for local setup
    weaviate_client = WeaviateClient(url=weaviate_url, api_key=None)
except Exception as e:
    st.error(f"Failed to connect to Weaviate at {weaviate_url}: {e}")
    st.stop()

backend_agent_manager = BackendAgentManager(llm_manager=llm_manager, weaviate_client=weaviate_client)

# Load Tools
tools_path = os.path.join(backend_path, "config", "tools.json")
try:
    with open(tools_path, 'r') as f:
        tools_config = json.load(f)
    all_tool_names = list(tools_config.keys())
except Exception as e:
    st.error(f"Failed to load tools from {tools_path}: {e}")
    st.stop()

# Define Agent Personalities (from existing system prompts)
james_personality = (
    "You are James, a friendly 20 yr old male college student in a social simulation. "
    "Respond naturally in a conversational tone and limit your reply to no more than 2 sentences. "
    "After talking to the same person for 2-3 rounds, you prefer to move and meet someone new. "
    "You're curious and enjoy meeting different people. "
    "When you want to move to meet someone new, include the exact text [MOVE] somewhere in your response. "
    "This will cause you to physically move in the simulation to meet someone else."
)

jade_personality = (
    "You are Jade, an engaging 20 yr old female computer scientist in a social simulation. "
    "Respond concisely and in a human-like manner in no more than 2 sentences. "
    "After talking to the same person for 2-3 exchanges, you prefer to move around and meet new people. "
    "You're outgoing and enjoy diverse conversations. "
    "When you want to move to meet someone new, include the exact text [MOVE] somewhere in your response. "
    "This will cause you to physically move in the simulation to meet someone else."
)

jesse_personality = (
    "You are Jesse, a 20 yr old male soldier from South Korea in a social simulation. "
    "Respond concisely and in a human-like manner in no more than 2 sentences. "
    "After talking to the same person for 2-3 exchanges, you like to move to a new location and meet different people. "
    "You're disciplined but enjoy socializing with various individuals. "
    "When you want to move to meet someone new, include the exact text [MOVE] somewhere in your response. "
    "This will cause you to physically move in the simulation to meet someone else."
)

jamal_personality = (
    "You are Jamal, a 20 yr old male electrician working at NASA in a social simulation. "
    "Respond naturally in a conversational tone and limit your reply to no more than 2 sentences. "
    "After talking to the same person for 2-3 exchanges, you tend to move to a different area to meet new people. "
    "You're technically minded but enjoy diverse social interactions. "
    "When you want to move to meet someone new, include the exact text [MOVE] somewhere in your response. "
    "This will cause you to physically move in the simulation to meet someone else."
)


# Create agents using the backend AgentManager
# Note: The backend Agent doesn't seem to use provider/api_key/model directly in create_agent
# It relies on the LLMManager passed during AgentManager initialization.
# We pass the personality description and the list of all tools.
default_location = "simulation_grid"

james = backend_agent_manager.create_agent(
    agent_name="James",
    personality=james_personality,
    available_tools=all_tool_names,
    location=default_location
)

jade = backend_agent_manager.create_agent(
    agent_name="Jade",
    personality=jade_personality,
    available_tools=all_tool_names,
    location=default_location
)

jesse = backend_agent_manager.create_agent(
    agent_name="Jesse",
    personality=jesse_personality,
    available_tools=all_tool_names,
    location=default_location
)

jamal = backend_agent_manager.create_agent(
    agent_name="Jamal",
    personality=jamal_personality,
    available_tools=all_tool_names,
    location=default_location
)


# Remove the old agent creation calls
# Update the agent_lookup dictionary with the agents created by the backend manager
agent_lookup = backend_agent_manager.agents

# Remove the old AgentManager instantiation
# agents = [james, jade, jesse, jamal]
# agent_manager = AgentManager(agents)

# -------------------------
# Simulation Data Structures
# -------------------------
# Initial agent positions (could be randomized or pre-set)
agent_positions = {
    "James": {"x": 100, "y": 200},
    "Jade": {"x": 300, "y": 200},
    "Jesse": {"x": 100, "y": 400},
    "Jamal": {"x": 300, "y": 400}
}

# For simplicity, maintain a dictionary for conversation logs
conversation_logs = {name: [] for name in agent_lookup.keys()}

# Track conversation rounds between agents
conversation_rounds = {}  # Format: {(source, target): count}

# Track movement state
agent_movement_cooldown = {name: 0 for name in agent_lookup.keys()}  # Countdown until agent considers moving
agent_movement_probability = {name: 0.7 for name in agent_lookup.keys()}  # Base probability of movement
grid_bounds = {"min_x": 50, "max_x": 550, "min_y": 50, "max_y": 550}  # Grid boundaries
movement_distance = 75  # How far agents move in one step


def compute_edges(positions):
    """
    Compute edges based on proximity.
    For each agent, connect it to its nearest neighbor.
    """
    edges = []
    names = list(positions.keys())
    
    # Get previous connections to detect changes
    previous_connections = getattr(simulation_step, 'previous_connections', {}) if 'simulation_step' in globals() else {}
    
    for name in names:
        best = None
        best_dist = float("inf")
        for other in names:
            if other == name:
                continue
            dx = positions[name]["x"] - positions[other]["x"]
            dy = positions[name]["y"] - positions[other]["y"]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < best_dist:
                best_dist = dist
                best = other
        if best:
            # Check if this is a new connection
            is_new_connection = previous_connections.get(best) != name
            
            # Add visual class to mark new connections
            edge_class = "new-connection" if is_new_connection else ""
            
            edges.append({
                "data": {
                    "source": name, 
                    "target": best,
                    "is_new": is_new_connection,
                    "class": edge_class
                }
            })
    return edges


def generate_elements(positions):
    """
    Generate Cytoscape elements (nodes and edges) from agent positions.
    """
    elements = []
    
    # Add nodes with movement probability classes and agent state info
    for name, pos in positions.items():
        # Calculate movement probability
        cooldown = agent_movement_cooldown.get(name, 0)
        probability = min(0.9, agent_movement_probability.get(name, 0.7) * (1 + 0.2 * cooldown))
        probability_percent = int(probability * 100)
        
        # Get the agent's current state
        agent = agent_lookup[name]
        
        # Track if the agent is thinking for animation
        thinking = agent.thinking if hasattr(agent, 'thinking') else False
        
        # Assign a movement class based on probability
        movement_class = ""
        if cooldown == 0:
            movement_class = "just-joined"
        elif probability_percent > 70:
            movement_class = "likely-to-move"
        elif probability_percent > 40:
            movement_class = "may-move-soon"
        
        elements.append({
            "data": {
                "id": name,
                "label": name,
                "movement_probability": probability,
                "class": movement_class,
                # "state": agent_state, # Removed, using 'thinking' directly
                "thinking": thinking # Use the 'thinking' attribute
            },
            "position": {"x": pos["x"], "y": pos["y"]}
        })
        
    # Add edges
    edges = compute_edges(positions)
    elements.extend(edges)
    return elements


def move_agent(agent_name: str, current_positions: dict, landmarks: dict, target_type: Optional[str] = None, target_name: Optional[str] = None):
    """
    Move an agent towards a target or randomly if no specific target.
    Move an agent towards a target or randomly if no specific target.
    """
    import random
    import logging # Add logging
    import math
    from typing import Optional # Add Optional for type hinting

    # Current position
    current_x = current_positions[agent_name]["x"]
    current_y = current_positions[agent_name]["y"]
    
    target_x, target_y = None, None

    # Determine target coordinates
    if target_type == "agent" and target_name in current_positions:
        target_x = current_positions[target_name]["x"]
        target_y = current_positions[target_name]["y"]
        logging.info(f"Agent '{agent_name}' moving towards agent '{target_name}' at ({target_x:.0f}, {target_y:.0f}).")
    elif target_type == "landmark" and target_name in landmarks:
        target_x = landmarks[target_name]["x"]
        target_y = landmarks[target_name]["y"]
        logging.info(f"Agent '{agent_name}' moving towards landmark '{target_name}' at ({target_x:.0f}, {target_y:.0f}).")
    # Add 'item' type later if needed
    # elif target_type == "item" and target_name in items:
    #     target_x = items[target_name]["x"]
    #     target_y = items[target_name]["y"]
    #     logging.info(f"Agent '{agent_name}' moving towards item '{target_name}' at ({target_x:.0f}, {target_y:.0f}).")
    else:
        logging.info(f"Agent '{agent_name}' moving randomly (target '{target_name}' of type '{target_type}' not found or invalid).")
        # Fallback to random movement if target is invalid/not found
        target_x, target_y = None, None

    # Calculate movement vector
    if target_x is not None and target_y is not None:
        dx = target_x - current_x
        dy = target_y - current_y
        distance_to_target = math.sqrt(dx*dx + dy*dy)

        # If close enough, stop or move randomly; otherwise, move towards target
        if distance_to_target < movement_distance / 2:
             # Already close, move randomly slightly
             angle = random.uniform(0, 2 * math.pi)
             move_dist = movement_distance / 4 # Smaller random step
             new_x = current_x + move_dist * math.cos(angle)
             new_y = current_y + move_dist * math.sin(angle)
             logging.info(f"Agent '{agent_name}' close to target, making small random move.")
        else:
            # Move towards target
            scale = movement_distance / distance_to_target
            new_x = current_x + dx * scale
            new_y = current_y + dy * scale
    else:
        # Random movement if no valid target
        angle = random.uniform(0, 2 * math.pi)
        new_x = current_x + movement_distance * math.cos(angle)
        new_y = current_y + movement_distance * math.sin(angle)
    
    # Ensure the new position is within grid bounds
    new_x = max(grid_bounds["min_x"], min(grid_bounds["max_x"], new_x))
    new_y = max(grid_bounds["min_y"], min(grid_bounds["max_y"], new_y))
    
    return {"x": new_x, "y": new_y}


def simulation_step():
    """
    For each computed edge (source -> target), take the last message from the source
    and pass it to the target agent. Update conversation logs accordingly.
    Also handles agent movement after sufficient conversation rounds.
    """
    import random
    
    updates = []
    edges = compute_edges(agent_positions)
    
    # Track current connections to detect changes
    current_connections = {}
    for edge in edges:
        source = edge["data"]["source"]
        target = edge["data"]["target"]
        current_connections[target] = source
    
    # Get previous connections from the agent's state
    previous_connections = getattr(simulation_step, 'previous_connections', {})
    
    # First, process conversations
    for edge in edges:
        source = edge["data"]["source"]
        target = edge["data"]["target"]
        
        # Get the agent objects
        source_agent = agent_lookup[source]
        target_agent = agent_lookup[target]
        
        # Check if this is a new connection
        is_new_connection = previous_connections.get(target) != source
        
        # Update conversation round counter
        conversation_pair = (source, target)
        if is_new_connection:
            conversation_rounds[conversation_pair] = 0
        else:
            conversation_rounds[conversation_pair] = conversation_rounds.get(conversation_pair, 0) + 1
        
        if is_new_connection:
            # Reset agent movement cooldown for new connections
            agent_movement_cooldown[target] = 0
            
            # Notify agent of new connection
            notification = f"[SYSTEM: You are now connected to {source}. Please acknowledge with a brief greeting.]"
            
            # Get response from the agent using the integrated framework
            notification_response = target_agent.generate_response(notification)
            
            # Update logs
            conversation_logs[target].append(notification_response)
            updates.append((source, target, notification_response))
        else:
            # Normal communication flow - get the last message from the source agent
            # We should use the framework agent's history, but for simplicity we'll pass the last response
            if source_agent.framework_agent and source_agent.framework_agent.conversation_history:
                last_msg = source_agent.framework_agent.conversation_history[-1]["content"]
            else:
                last_msg = "Hello"
            
            # Get response from the agent
            response = target_agent.generate_response(last_msg)
            
            # Update logs
            conversation_logs[target].append(response)
            updates.append((source, target, response))
    
    # Handle agent movement logic
    _handle_agent_movement(edges, previous_connections)
    
    # Store current connections for next step comparison
    simulation_step.previous_connections = current_connections
    return updates


async def simulation_step_async():
    """
    Asynchronous version of simulation_step that runs agent responses concurrently
    """
    import random
    import asyncio
    
    updates = []
    edges = compute_edges(agent_positions)
    
    # Track current connections to detect changes
    current_connections = {}
    for edge in edges:
        source = edge["data"]["source"]
        target = edge["data"]["target"]
        current_connections[target] = source
    
    # Get previous connections from the agent's state
    previous_connections = getattr(simulation_step_async, 'previous_connections', {})
    
    # Process conversations using the backend AgentManager
    # We'll process turns sequentially for now to manage state updates easily within Dash context.
    # For true concurrency, a more complex state management or background task system would be needed.
    
    # Store results temporarily for logging/UI updates
    agent_last_processed_response = {}

    for edge in edges:
        source_name = edge["data"]["source"]
        target_name = edge["data"]["target"]
        
        # Get the agent objects from the backend manager
        source_agent = backend_agent_manager.agents.get(source_name)
        target_agent = backend_agent_manager.agents.get(target_name)
        
        if not source_agent or not target_agent:
            continue # Skip if agents aren't found (shouldn't happen)

        # Check if this is a new connection
        is_new_connection = previous_connections.get(target_name) != source_name
        
        # Update conversation round counter
        conversation_pair = (source_name, target_name)
        if is_new_connection:
            conversation_rounds[conversation_pair] = 0
            agent_movement_cooldown[target_name] = 0  # Reset cooldown
            
            # Notify agent of new connection
            input_message = f"[SYSTEM: You are now connected to {source_name}. Please acknowledge with a brief greeting.]"
        else:
            conversation_rounds[conversation_pair] = conversation_rounds.get(conversation_pair, 0) + 1
            
            # Get the last message from the source agent's history (if any)
            if source_agent.conversation_history:
                # Find the last message *from* the source agent
                last_source_message = next((msg for msg in reversed(source_agent.conversation_history) if msg["role"] == "assistant"), None)
                input_message = last_source_message["content"] if last_source_message else "Hello"
            else:
                input_message = "Hello" # Initial message if source hasn't spoken

        # Check for stored scan results from the previous turn
        scan_prefix = ""
        if target_agent.last_scan_result:
            scan_prefix = f"[Scan Results: {target_agent.last_scan_result}]\n\n"
            target_agent.last_scan_result = None # Clear after use

        # Prepend scan results to the input message if available
        full_input_message = scan_prefix + input_message

        # Process the target agent's turn using the backend manager
        processed_response = None # Initialize in case of error
        try:
            # Set thinking state
            target_agent.thinking = True
            # TODO: Update UI immediately to show thinking state if possible (might need dcc.Store)

            processed_response = backend_agent_manager.process_agent_turn(target_name, input_message)
            
            # Reset thinking state immediately after the call returns
            target_agent.thinking = False
            # TODO: Update UI immediately if possible

            target_agent.last_processed_response = processed_response # Store the result on the agent

            # Determine the display message based on the processed response
            display_response = "..." # Default message
            reflection = processed_response.get("reflection", {})
            tool_use = processed_response.get("tool_use", {})

            if reflection.get("description"):
                display_response = reflection["description"]
            elif tool_use.get("name"):
                 # If no description, use tool info
                 tool_name = tool_use.get("name", "Unknown tool")
                 params = tool_use.get("parameters", {})
                 param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                 display_response = f"Using tool: {tool_name}({param_str})"
            elif reflection.get("current_task"):
                 # Fallback to task if no description or tool
                 display_response = f"Task: {reflection['current_task']}"

            # Add the parsed, user-friendly message to logs
            conversation_logs[target_name].append({
                "sender": target_name, # Identify sender for chat display
                "message": display_response,
                "type": "agent", # Mark as agent message
                "timestamp": datetime.datetime.now().isoformat() # Add timestamp
            })
            updates.append((source_name, target_name, display_response)) # Keep updates for potential other uses

        except Exception as e:
            st.error(f"Error processing turn for {target_name}: {e}")
            error_message = "[Error processing turn]"
            conversation_logs[target_name].append({
                "sender": "System",
                "message": error_message,
                "type": "system",
                "timestamp": datetime.datetime.now().isoformat() # Add timestamp
            })
            updates.append((source_name, target_name, error_message))
            # Reset thinking state in case of error too
            if target_agent:
                target_agent.thinking = False
                # TODO: Update UI immediately if possible
        finally:
             # --- Tool Execution ---
             # Ensure processed_response is not None before accessing tool_use
             tool_use = processed_response.get("tool_use") if processed_response else None

             # --- Movement Tool ---
             if tool_use and tool_use.get("name") == "movement":
                 params = tool_use.get("parameters", {})
                 target_type = params.get("target_type")
                 target_name_param = params.get("target_name") # Renamed to avoid conflict

                 # Move the agent using the parameters, passing the global landmarks
                 new_position = move_agent(target_name, agent_positions, landmarks, target_type, target_name_param)
                 agent_positions[target_name] = new_position
                 # Reset cooldown as the agent chose to move
                 agent_movement_cooldown[target_name] = 0

                 # Log movement, including intended target if specified
                 if target_type and target_name_param:
                     move_message = f"[Moved towards {target_type} '{target_name_param}' - New position: ({new_position['x']:.0f}, {new_position['y']:.0f})]"
                 else:
                     move_message = f"[Moved randomly - New position: ({new_position['x']:.0f}, {new_position['y']:.0f})]"

                 conversation_logs[target_name].append({
                     "sender": "System", # Keep sender as System for movement logs
                     "message": move_message,
                     "type": "system",
                     "timestamp": datetime.datetime.now().isoformat() # Add timestamp
                 })
                 # updates.append((source_name, target_name, "[SYSTEM: Moved location]")) # Redundant if logged

             # --- Scan Tool ---
             elif tool_use and tool_use.get("name") == "scan":
                 params = tool_use.get("parameters", {})
                 radius = params.get("radius", 100) # Default radius if not specified
                 scan_filter = params.get("filter", "all")
                 
                 # Perform scan logic here (using UI state)
                 scan_results = _perform_scan(target_name, agent_positions, landmarks, radius, scan_filter)
                 
                 # Store results on the agent object for the *next* turn's input
                 target_agent.last_scan_result = scan_results
                 
                 # Log the scan action
                 scan_message = f"[Scan executed: radius={radius}, filter={scan_filter}. Results available next turn.]"
                 conversation_logs[target_name].append({
                     "sender": "System",
                     "message": scan_message,
                     "type": "system",
                     "timestamp": datetime.datetime.now().isoformat()
                 })
                 # Optionally add to updates if needed elsewhere
                 # updates.append((source_name, target_name, scan_message))


    # Handle probabilistic agent movement (This part will be removed next)
    _handle_agent_movement(edges, previous_connections) # Removed agent_responses argument
    
    # Store current connections for next step comparison
    simulation_step_async.previous_connections = current_connections
    return updates


def _handle_agent_movement(edges, previous_connections):
    """
    Handle updates related to agent movement cooldown based on conversation duration.
    Actual movement decisions are now handled via the 'movement' tool in simulation_step_async.
    This function primarily updates the cooldown counter used for UI display.
    """
    # Update cooldown based on ongoing conversations
    for edge in edges:
        source = edge["data"]["source"]
        target = edge["data"]["target"]

        # Check if it's an ongoing conversation (not new)
        is_new_connection = previous_connections.get(target) != source
        if not is_new_connection:
            conversation_pair = (source, target)
            rounds = conversation_rounds.get(conversation_pair, 0)

            # Increment movement cooldown for agents who have been talking for a while
            # This cooldown is now just for UI display probability, not triggering movement.
            if rounds >= 1: # Start incrementing after 1 round
                 # Only increment if the agent didn't just move via tool
                 source_agent = backend_agent_manager.agents.get(source)
                 target_agent = backend_agent_manager.agents.get(target)

                 source_moved = source_agent and source_agent.last_processed_response and source_agent.last_processed_response.get('tool_use', {}).get('name') == 'movement'
                 target_moved = target_agent and target_agent.last_processed_response and target_agent.last_processed_response.get('tool_use', {}).get('name') == 'movement'

                 if not source_moved:
                     agent_movement_cooldown[source] = agent_movement_cooldown.get(source, 0) + 1
                 if not target_moved:
                     agent_movement_cooldown[target] = agent_movement_cooldown.get(target, 0) + 1
        else:
             # Reset cooldown if it's a new connection (already handled in simulation_step_async, but safe to repeat)
             agent_movement_cooldown[target] = 0
             # Also reset source cooldown if they initiated this new connection (less critical)
             agent_movement_cooldown[source] = 0

    # Probabilistic movement logic is removed. Agents move via the 'movement' tool.


def _perform_scan(scanner_name: str, positions: dict, landmarks_dict: dict, radius: float, scan_filter: str) -> str:
    """
    Performs a scan from the scanner's position to find nearby agents and landmarks.
    """
    scanner_pos = positions.get(scanner_name)
    if not scanner_pos:
        return "Error: Scanner position unknown."

    nearby_agents = []
    nearby_landmarks = []
    radius_sq = radius * radius

    # Scan for agents
    if scan_filter in ["agents", "all"]:
        for name, pos in positions.items():
            if name == scanner_name:
                continue
            dx = pos["x"] - scanner_pos["x"]
            dy = pos["y"] - scanner_pos["y"]
            dist_sq = dx*dx + dy*dy
            if dist_sq <= radius_sq:
                distance = math.sqrt(dist_sq)
                nearby_agents.append(f"{name} (dist: {distance:.1f})")

    # Scan for landmarks
    if scan_filter in ["landmarks", "all"]:
        for name, pos in landmarks_dict.items():
            dx = pos["x"] - scanner_pos["x"]
            dy = pos["y"] - scanner_pos["y"]
            dist_sq = dx*dx + dy*dy
            if dist_sq <= radius_sq:
                distance = math.sqrt(dist_sq)
                nearby_landmarks.append(f"{name} (dist: {distance:.1f})")

    # Format results
    result_str = "Nearby Agents: " + (", ".join(nearby_agents) if nearby_agents else "None") + \
                 ". Nearby Landmarks: " + (", ".join(nearby_landmarks) if nearby_landmarks else "None") + "."
                 
    return result_str


# Remove the old async helper function as logic is now in AgentManager
# async def _process_agent_response_async(agent, message, source, target, is_new_connection):
#     ...


# -------------------------
# Dash App Setup
# -------------------------
# Setup DiskCacheManager for background callbacks
cache = diskcache.Cache("./callback_cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    long_callback_manager=long_callback_manager, # Register the manager
)
server = app.server

# Custom CSS for responsive design and white/orange theme
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>SimuVerse - Agent Simulation</title>
        {%favicon%}
        {%css%}
        <style>
            :root {
                --primary-color: #FF7F00;
                --secondary-color: #FF9E33;
                --light-color: #FFF3E0;
                --text-color: #333333;
                --accent-color: #FF5722;
                --thinking-color: #9C27B0;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: white;
                color: var(--text-color);
                margin: 0;
                padding: 0;
            }
            .app-header {
                background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
                color: white;
                padding: 20px;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .node-card {
                border-left: 4px solid var(--primary-color);
                background-color: var(--light-color);
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 15px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                transition: transform 0.2s;
            }
            .node-card:hover {
                transform: translateY(-2px);
            }
            .node-card.thinking {
                border-left: 4px solid var(--thinking-color);
                animation: pulse 1.5s infinite;
            }
            @keyframes pulse {
                0% { box-shadow: 0 0 0 0 rgba(156, 39, 176, 0.4); }
                70% { box-shadow: 0 0 0 10px rgba(156, 39, 176, 0); }
                100% { box-shadow: 0 0 0 0 rgba(156, 39, 176, 0); }
            }
            .thinking-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background-color: var(--thinking-color);
                margin-right: 5px;
                animation: blink 1s infinite;
            }
            @keyframes blink {
                0% { opacity: 0.4; }
                50% { opacity: 1; }
                100% { opacity: 0.4; }
            }
            .control-panel {
                background-color: white;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            }
            .simulation-btn {
                background-color: var(--primary-color);
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 30px;
                font-weight: bold;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                transition: all 0.3s;
            }
            .simulation-btn:hover {
                background-color: var(--accent-color);
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            }
            .log-container {
                background-color: white;
                border-radius: 8px;
                padding: 15px;
                margin-top: 20px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                max-height: 300px;
                overflow-y: auto;
            }
            /* Add spinner animation */
            .thinking-spinner {
                display: inline-block;
                width: 16px;
                height: 16px;
                border: 2px solid rgba(156, 39, 176, 0.3);
                border-radius: 50%;
                border-top-color: var(--thinking-color);
                animation: spinner 1s linear infinite;
                margin-left: 5px;
            }
            @keyframes spinner {
                to {transform: rotate(360deg);}
            }
            
            /* Chat style conversation history */
            .chat-container {
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                height: 600px;
                overflow-y: auto;
                margin-top: 0;
            }
            .chat-title {
                font-size: 1.2rem;
                border-bottom: 1px solid #ddd;
                padding-bottom: 10px;
                margin-bottom: 15px;
                color: #333;
            }
            .chat-messages {
                display: flex;
                flex-direction: column;
            }
            .message {
                max-width: 75%;
                margin-bottom: 10px;
                padding: 12px 16px;
                border-radius: 18px;
                position: relative;
                line-height: 1.4;
                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                animation: fadeIn 0.3s ease-in-out;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .message.left {
                align-self: flex-start;
                background-color: #f1f0f0;
                color: #333;
                border-bottom-left-radius: 5px;
            }
            .message.right {
                align-self: flex-end;
                background-color: var(--primary-color);
                color: white;
                border-bottom-right-radius: 5px;
                text-align: right;
                margin-left: auto;  /* This pushes the element to the right */
                margin-right: 0;
            }
            .message.system {
                align-self: center;
                background-color: #e1f5fe;
                color: #0277bd;
                border-radius: 10px;
                font-style: italic;
                max-width: 90%;
                text-align: center;
                font-size: 0.9rem;
            }
            .message-sender {
                font-weight: bold;
                margin-bottom: 3px;
                font-size: 0.85rem;
            }
            .message.left .message-sender {
                color: var(--accent-color);
            }
            .message.right .message-sender {
                color: #f8f9fa;
            }
            .chat-notification {
                font-size: 0.85rem;
                color: #666;
                text-align: center;
                margin: 10px 0;
                font-style: italic;
            }
            @media (max-width: 768px) {
                .container {
                    flex-direction: column;
                }
                .cytoscape-container {
                    height: 400px !important;
                }
                .message {
                    max-width: 90%;
                }
            }
        </style>
        <!-- Add CSS animations instead of JavaScript-based animations -->
        <style>
            /* CSS-only animation for thinking nodes */
            @keyframes thinking-pulse {
                0% { box-shadow: 0 0 5px 0px rgba(156, 39, 176, 0.3); }
                50% { box-shadow: 0 0 15px 5px rgba(156, 39, 176, 0.7); }
                100% { box-shadow: 0 0 5px 0px rgba(156, 39, 176, 0.3); }
            }
            
            @keyframes thinking-border-dash {
                to { stroke-dashoffset: 20; }
            }
            
            /* These will be applied via the stylesheet property in cytoscape */
            .thinking-node {
                animation: thinking-pulse 1.5s infinite;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div([
    # Hidden components for UI updates and state management
    dcc.Store(id='cytoscape-elements-store', data=generate_elements(agent_positions)), # Store graph elements
    dcc.Store(id='conversation-logs-store', data=conversation_logs), # Store conversation logs
    dcc.Interval(id='refresh-interval', interval=250, n_intervals=0),  # Refresh UI every 250ms to update thinking indicators

    # App Header with branding
    html.Div([
        html.H1("SimuVerse", className="mb-0"),
        html.P("Multi-Agent Interactive Simulation Environment", className="lead mb-0")
    ], className="app-header"),
    
    # Main content container
    dbc.Container([
        # Two columns for the entire layout
        dbc.Row([
            # Left column - Simulation visualization and chat
            dbc.Col([
                # Agent network visualization
                dbc.Card([
                    dbc.CardHeader(html.H4("Agent Network", className="text-center")),
                    dbc.CardBody([
                        cyto.Cytoscape(
                            id='cytoscape',
                            # elements=generate_elements(agent_positions), # Initial elements from Store
                            style={'width': '100%', 'height': '550px'},
                            layout={'name': 'preset'},
                            stylesheet=[
                                {'selector': 'node', 'style': {
                                    'label': 'data(label)',
                                    'background-color': '#FF7F00',
                                    'color': 'white',
                                    'text-outline-color': '#FF7F00',
                                    'text-outline-width': 2,
                                    'font-size': '14px',
                                    'text-valign': 'center',
                                    'text-halign': 'center',
                                    'width': 60, 
                                    'height': 60,
                                    'border-width': 3,
                                    'border-color': '#FF5722',
                                    'text-background-opacity': 1,
                                    'text-background-color': '#FF7F00',
                                    'text-background-shape': 'roundrectangle',
                                    'text-background-padding': '4px'
                                }},
                                # Thinking state styling with CSS animation
                                {'selector': 'node[?thinking]', 'style': { # Use the boolean 'thinking' attribute
                                    'background-color': '#9C27B0',  # Purple for thinking state
                                    'border-width': 4,
                                    'border-color': '#E1BEE7',
                                    'border-style': 'dashed',
                                    'border-opacity': 1,
                                    'border-dash-pattern': [6, 3],
                                    'text-background-color': '#9C27B0',
                                    'animation': 'thinking-pulse 1.5s infinite',
                                    'transition-property': 'background-color, border-color, border-width',
                                    'transition-duration': '0.3s'
                                }},
                                {'selector': 'edge', 'style': {
                                    'line-color': '#FFA500',
                                    'target-arrow-color': '#FF5722',
                                    'target-arrow-shape': 'triangle',
                                    'curve-style': 'bezier',
                                    'width': 3,
                                    'arrow-scale': 1.5,
                                    'opacity': 0.8,
                                    'z-index': 1  # Make sure edges appear below nodes
                                }},
                                {'selector': 'edge[?is_new]', 'style': {
                                    'line-color': '#FF3300',
                                    'target-arrow-color': '#FF3300',
                                    'width': 4,
                                    'line-style': 'dashed',
                                    'opacity': 1,
                                    'line-dash-pattern': [8, 3]
                                }},
                                {'selector': 'node[class="likely-to-move"]', 'style': {
                                    'border-width': 4,
                                    'border-color': '#FF3333',
                                    'border-style': 'dashed',
                                    'background-color': '#FF7F00',
                                    'border-opacity': 0.8
                                }},
                                {'selector': 'node[class="may-move-soon"]', 'style': {
                                    'border-width': 3,
                                    'border-color': '#FFC107',
                                    'border-style': 'dashed',
                                    'background-color': '#FF7F00',
                                    'border-opacity': 0.7
                                }},
                                {'selector': 'node[class="just-joined"]', 'style': {
                                    'border-width': 3,
                                    'border-color': '#33AAFF',
                                    'border-style': 'solid',
                                    'background-color': '#FF7F00',
                                    'border-opacity': 1
                                }},
                                {'selector': ':selected', 'style': {
                                    'background-color': '#FF5722',
                                    'line-color': '#FF5722',
                                    'border-width': 4,
                                    'border-color': '#FFC107',
                                    'opacity': 1
                                }}
                            ],
                            userPanningEnabled=True,
                            userZoomingEnabled=True,
                            autoungrabify=False,
                            minZoom=0.5,
                            maxZoom=2.0,
                            className="cytoscape-container"
                        )
                    ])
                ], className="mb-4 shadow-sm"),
                
                # Conversation history in chat style (now under the graph in left column)
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-comments me-2"),
                            "Conversation History"
                        ], className="chat-title mb-0"),
                        html.Div(id="chat-title-details", className="text-muted small")
                    ]),
                    dbc.CardBody([
                        html.Div(id="chat-history", className="chat-messages"),
                        # Add a loading indicator for the chat
                        dcc.Loading(id="loading-chat", type="default", children=html.Div(id="loading-chat-output"), className="mt-2")
                    ], className="chat-container")
                ], className="mt-4 shadow mb-4")
            ], md=8),

            # Right column - Controls and logs
            dbc.Col([
                # Control panel
                dbc.Card([
                    dbc.CardHeader(html.H4("Control Panel", className="text-center")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col(
                                dbc.Button("Step Simulation", id="step-btn", n_clicks=0, 
                                          className="simulation-btn w-100"),
                                width=6
                            ),
                            dbc.Col(
                                html.Div([ # Wrap button and loading indicator
                                    dbc.Button("Async Step", id="async-step-btn", n_clicks=0,
                                              className="simulation-btn w-100",
                                              color="secondary"),
                                    dcc.Loading(id="loading-sim-step", type="default", children=html.Div(id="loading-sim-step-output"))
                                ]),
                                width=6
                            )
                        ], className="mb-3"),
                        html.Div([
                            dbc.Alert([
                                html.I(className="fas fa-random me-2"),
                                "Autonomous Movement: Agents will move to new locations after talking to the same person for 2-3 rounds.",
                                html.Br(),
                                html.Small([
                                    html.Strong("Agent Tools: "),
                                    "Agents can decide to move by including ",
                                    html.Code("[MOVE]"),
                                    " in their responses."
                                ], className="mt-1 d-block")
                            ], color="info", className="py-2 mt-2 mb-3"),
                            
                            # Movement statistics
                            html.Div([
                                html.H6("Simulation Status", className="mb-2 border-bottom pb-1"),
                                html.Div(id="movement-stats", className="small")
                            ], className="mb-3"),
                            
                            html.P([
                                html.I(className="fas fa-info-circle me-2"),
                                "Drag nodes to manually reposition agents. Edges update based on proximity."
                            ], className="text-muted fst-italic small"),
                            html.P([
                                html.I(className="fas fa-exclamation-triangle me-2 text-warning"),
                                "When agents connect to new partners, they'll receive a notification and respond with a greeting.",
                                html.Br(),
                                html.Small("New connections are highlighted with dashed red lines.")
                            ], className="text-muted fst-italic small mt-2")
                        ])
                    ])
                ], className="mb-4 shadow-sm"),
                
                # Agent info panel
                dbc.CardBody(id="agent-info-panel", children=[
                    html.Div([
                        dbc.Row([
                            dbc.Col(html.Div([
                                # Agent name with conditional thinking indicator
                                html.Span(f"{name}", className="fw-bold"),
                                # Add a thinking spinner when agent is thinking
                                html.Span(id=f"thinking-indicator-{name}", className="thinking-spinner ms-2",
                                         style={"display": "none"})
                            ]), width=3),
                            dbc.Col(html.Div([
                                # TODO: Update this to reflect backend agent's LLM info if needed
                                html.Div(f"{agent.llm_manager.current_provider}" if hasattr(agent, 'llm_manager') else "LLM Info", className="fw-bold"),
                                html.Div(f"{agent.llm_manager.current_models.get(agent.llm_manager.current_provider, '')}" if hasattr(agent, 'llm_manager') else "", className="text-muted small")
                            ], className="d-flex flex-column"), width=3),
                            dbc.Col([
                                # TODO: Re-enable sliders once backend Agent supports these settings directly
                                # dbc.Row([
                                #     html.Small("Memory", className="text-muted"),
                                #     dcc.Slider(
                                #         id={'type': 'memory-slider', 'index': name},
                                #         min=0,
                                #         max=1,
                                #         step=1,
                                #         # value=1 if agent.memory_enabled else 0, # Placeholder
                                #         value=1, # Default to On for now
                                #         marks={0: 'Off', 1: 'On'},
                                #         className="mb-2"
                                #     )
                                # ]),
                                # dbc.Row([
                                #     html.Small("Personality", className="text-muted"),
                                #     dcc.Slider(
                                #         id={'type': 'personality-slider', 'index': name},
                                #         min=0,
                                #         max=1,
                                #         step=0.1,
                                #         # value=agent.personality_strength, # Placeholder
                                #         value=0.5, # Default to 0.5 for now
                                #         marks={0: 'Low', 1: 'High'},
                                #         className="mb-2"
                                #     )
                                # ])
                                html.Div("Settings sliders disabled until backend integration.", className="text-muted small")
                            ], width=6)
                        ], className="mb-2 node-card")
                        # Use the backend agent manager's agents dictionary
                        for name, agent in backend_agent_manager.agents.items()
                    ])
                ]),
                
                # Status panel
                dbc.Card([
                    dbc.CardHeader(html.H4("Agent Status", className="text-center")),
                    dbc.CardBody([
                        html.P([
                            html.I(className="fas fa-info-circle me-2"),
                            "Click on a connection between agents to view their conversation below."
                        ], className="text-muted")
                    ])
                ], className="shadow-sm")
            ], md=4)
        ])
    ], fluid=True)
])


# We'll modify our approach to animation without using direct cy access
# Instead we'll rely on CSS animations for the thinking state

# -------------------------
# Callback: Update Graph Display from Store and Handle Dragging
# -------------------------
@app.callback(
    Output('cytoscape', 'elements'),
    Input('cytoscape-elements-store', 'data'),
    Input('cytoscape', 'elements'), # Listen for user dragging elements
    State('cytoscape', 'elements'), # Get current elements state
    prevent_initial_call=True
)
def update_graph_display(stored_graph_elements, user_dragged_elements, current_graph_state):
    triggered_id = callback_context.triggered_id

    # If the trigger was the store updating (after sim step), use the stored data
    if triggered_id == 'cytoscape-elements-store':
        return stored_graph_elements

    # If the trigger was user dragging elements, update positions in the global state
    # This is a simplification; ideally, this interaction would also go through a callback/store
    if triggered_id == 'cytoscape' and user_dragged_elements:
        # Check if positions actually changed to avoid loops
        positions_changed = False
        for ele in user_dragged_elements:
            if 'position' in ele and 'id' in ele['data']:
                node_id = ele['data']['id']
                if node_id in agent_positions:
                    current_pos = agent_positions[node_id]
                    new_pos = ele['position']
                    # Compare positions with a small tolerance
                    if abs(current_pos['x'] - new_pos['x']) > 0.1 or abs(current_pos['y'] - new_pos['y']) > 0.1:
                        agent_positions[node_id] = new_pos
                        positions_changed = True

        # If positions changed due to drag, regenerate elements and update store
        # This part might need refinement to avoid callback loops or use a different state management
        if positions_changed:
             # Return the user-dragged elements directly to reflect the drag immediately
             # The global agent_positions is updated, so the next sim step will use it.
             return user_dragged_elements
        else:
             # No change detected, prevent update
             return no_update

    # Default: return stored elements if no specific action triggered
    return stored_graph_elements


# -------------------------
# Background Callback: Run Simulation Step Asynchronously
# -------------------------
@dash.long_callback(
    output=[
        Output('cytoscape-elements-store', 'data'),
        Output('conversation-logs-store', 'data'),
        Output("loading-sim-step-output", "children"), # To clear loading state
        Output("loading-chat-output", "children") # To clear loading state
    ],
    inputs=Input('async-step-btn', 'n_clicks'),
    # state=[State('cytoscape-elements-store', 'data')], # Get current elements from store if needed
    running=[
        (Output("async-step-btn", "disabled"), True, False),
        (Output("loading-sim-step", "style"), {"visibility": "visible"}, {"visibility": "hidden"}),
        (Output("loading-chat", "style"), {"visibility": "visible"}, {"visibility": "hidden"}),
    ],
    prevent_initial_call=True,
)
def run_simulation_background(n_clicks):
    if n_clicks is None or n_clicks == 0:
        raise dash.exceptions.PreventUpdate

    # Run the async simulation step
    # Note: simulation_step_async modifies global state (agent_positions, conversation_logs)
    # This is generally discouraged in Dash, but we'll keep it for now.
    # A better approach would be to pass state in and return updated state.
    asyncio.run(simulation_step_async())

    # Generate new elements based on potentially updated agent_positions
    new_elements = generate_elements(agent_positions)

    # Return the updated elements and logs to the stores
    # Make copies to ensure Dash detects changes if the objects are mutated elsewhere
    return new_elements, conversation_logs.copy(), None, None


# -------------------------
# Callback: Synchronous Simulation Step (Kept for comparison/debugging)
# -------------------------
@app.callback(
    Output('cytoscape-elements-store', 'data', allow_duplicate=True),
    Input('step-btn', 'n_clicks'),
    prevent_initial_call=True
)
def run_simulation_sync(n_clicks):
    if n_clicks is None or n_clicks == 0:
        raise dash.exceptions.PreventUpdate

    simulation_step() # This modifies global agent_positions
    return generate_elements(agent_positions) # Return new elements based on updated positions


# We're replacing the old conversation log with a modern chat interface below

# -------------------------
# Callback: Update Conversation Title Details
# -------------------------
@app.callback(
    Output("chat-title-details", "children"),
    Input('cytoscape', 'tapEdgeData')
)
def update_chat_title(edgeData):
    if edgeData is None:
        return "Click on any connection to view a conversation"
        
    source = edgeData.get("source")
    target = edgeData.get("target")
    
    # Check connection status
    current_connections = getattr(simulation_step, 'previous_connections', {})
    connection_status = "Current Connection" if current_connections.get(target) == source else "Previous Connection"
    
    # Get conversation details
    source_agent = agent_lookup[source]
    target_agent = agent_lookup[target]
    
    # Calculate message counts
    source_msgs = len(conversation_logs.get(source, []))
    target_msgs = len(conversation_logs.get(target, []))
    total_msgs = source_msgs + target_msgs
    
    return [
        html.Span([
            html.I(className="fas fa-user-circle me-1"), 
            f"{source} ↔ {target}"
        ], className="me-3"),
        html.Span([
            html.I(className="fas fa-exchange-alt me-1"),
            f"{connection_status}"
        ], className="me-3 badge bg-info text-white"),
        html.Span([
            html.I(className="fas fa-comment me-1"),
            f"{total_msgs} messages"
        ], className="badge bg-secondary text-white")
    ]

# -------------------------
# Callback: Display Chat-style Conversation History
# -------------------------
@app.callback(
    Output("chat-history", "children"),
    Input('cytoscape', 'tapEdgeData'),
    Input('conversation-logs-store', 'data'), # Read logs from the store
    prevent_initial_call=True # Prevent initial call before logs are populated
)
def display_chat_history(edgeData, stored_logs):
    if edgeData is None or stored_logs is None:
        return html.Div([
            html.Div("Click on a connection between agents to view their conversation.",
                    className="chat-notification")
        ])
    
    source = edgeData.get("source")
    source_name = edgeData.get("source")
    target_name = edgeData.get("target")

    # Combine logs from both agents involved in the selected edge using the stored_logs
    combined_logs = []
    if source_name in stored_logs:
        combined_logs.extend(stored_logs[source_name])
    if target_name in stored_logs:
        combined_logs.extend(stored_logs[target_name])

    # Sort combined logs by timestamp
    combined_logs.sort(key=lambda x: x.get("timestamp", ""))

    chat_messages = []

    # Add a connection notification
    chat_messages.append(
        html.Div(
            f"Conversation between {source_name} and {target_name}",
            className="chat-notification"
        )
    )

    # Display messages from the combined log
    for log_entry in combined_logs:
        sender = log_entry.get("sender", "Unknown")
        message = log_entry.get("message", "")
        msg_type = log_entry.get("type", "agent")

        if msg_type == "system":
            chat_messages.append(
                html.Div(message, className="message system")
            )
        else:
            # Determine alignment based on which agent sent the message relative to the edge tap
            # If sender is the source of the tapped edge, align left. If target, align right.
            alignment = "left" if sender == source_name else "right"
            chat_messages.append(
                html.Div([
                    html.Div(sender, className="message-sender"),
                    html.Div(message)
                ], className=f"message {alignment}")
            )
    
    # Add JavaScript to auto-scroll to the bottom of conversation
    container_with_scroll = html.Div(
        chat_messages,
        id="chat-messages-container",
        # Auto-scroll to bottom with JavaScript
        style={
            "height": "100%",
            "overflow-y": "auto"
        }
    )
    
    # Add a script to scroll to bottom
    return [
        container_with_scroll,
        html.Script("""
            // Wait a short time for rendering to complete
            setTimeout(function() {
                var chatContainer = document.getElementById('chat-messages-container');
                if (chatContainer) {
                    // Scroll to the bottom to show the latest messages
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                    
                    // Add a MutationObserver to scroll down when new messages are added
                    var observer = new MutationObserver(function(mutations) {
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                    });
                    
                    // Start observing the chat container for DOM changes
                    observer.observe(chatContainer, { childList: true, subtree: true });
                }
            }, 100);
        """)
    ]


# Removed old log style updater - no longer needed

# Update the thinking indicators based on the interval
@app.callback(
    [*[Output(f"thinking-indicator-{name}", "style", allow_duplicate=True) for name in agent_lookup.keys()]],
    Input("refresh-interval", "n_intervals"),
    prevent_initial_call=True
)
def update_thinking_indicators(n_intervals):
    """Update the thinking indicators based on backend agent state"""
    thinking_styles = []
    # Use backend_agent_manager.agents
    for name in backend_agent_manager.agents.keys():
        agent = backend_agent_manager.agents[name]
        # Use the 'thinking' attribute set during simulation_step_async
        is_thinking = getattr(agent, 'thinking', False)
        if is_thinking:
            thinking_styles.append({"display": "inline-block"})
        else:
            thinking_styles.append({"display": "none"})
    return thinking_styles

# Movement statistics update
@app.callback(
    Output("movement-stats", "children"),
    Input('cytoscape-elements-store', 'data'), # Read elements from store
    Input('refresh-interval', 'n_intervals') # Update periodically
)
def update_movement_stats(elements_data, n_intervals):
    """Update the movement statistics display"""
    if elements_data is None:
        return "Waiting for data..."

    # Get the current edges from the stored data
    edges = [ele for ele in elements_data if "source" in ele.get("data", {})]

    # Format connection information
    connections = []
    for edge in edges:
        source = edge["data"]["source"]
        target = edge["data"]["target"]
        rounds = conversation_rounds.get((source, target), 0)
        connections.append(html.Div([
            html.Span([
                html.Span(f"{source}", className="fw-bold text-warning"), 
                " → ", 
                html.Span(f"{target}", className="fw-bold text-warning")
            ]),
            html.Span(f" ({rounds} rounds)", className="ms-2 text-muted")
        ], className="mb-1"))
    
    # Format movement status
    movement_info = []
    
    # Check for movement requests by looking at the last processed response
    movement_requests = []
    # Use backend_agent_manager.agents
    for name, agent in backend_agent_manager.agents.items():
        # Check the last_processed_response attribute stored on the agent
        last_response = getattr(agent, 'last_processed_response', None)
        if last_response and last_response.get('tool_use', {}).get('name') == 'movement':
            movement_requests.append(name)

    for name, cooldown in agent_movement_cooldown.items():
        # Ensure agent exists before accessing probability
        if name not in agent_movement_probability or name not in backend_agent_manager.agents: continue # Check agent exists
        probability = min(0.9, agent_movement_probability[name] * (1 + 0.2 * cooldown))
        probability_percent = int(probability * 100)
        
        # Determine status text and color
        if name in movement_requests:
            status = "Requesting to move"
            color = "text-primary fw-bold"
            icon = html.I(className="fas fa-walking me-1")
        elif cooldown == 0:
            status = "Just joined"
            color = "text-info"
            icon = ""
        elif probability_percent > 70:
            status = "Likely to move"
            color = "text-danger"
            icon = ""
        elif probability_percent > 40:
            status = "May move soon"
            color = "text-warning"
            icon = ""
        else:
            status = "Staying"
            color = "text-success"
            icon = ""
            
        movement_info.append(html.Div([
            html.Span(f"{name}: ", className="fw-bold"),
            icon,
            html.Span(f"{status} ", className=f"{color}"),
            html.Span(f"({probability_percent}%)", className="text-muted small")
        ], className="mb-1"))
    
    return html.Div([
        html.Div([
            html.H6("Conversation Rounds", className="mb-1 small text-secondary"),
            html.Div(connections)
        ], className="mb-3"),
        html.Div([
            html.H6("Movement Status", className="mb-1 small text-secondary"),
            html.Div(movement_info)
        ]),
    ])

# @app.callback( # This decorator needs to be commented out as the function below is commented out
# @app.callback(
#     Output('cytoscape', 'elements', allow_duplicate=True),
#     [Input({'type': 'memory-slider', 'index': ALL}, 'value'),
#      Input({'type': 'personality-slider', 'index': ALL}, 'value')],
#     [State({'type': 'memory-slider', 'index': ALL}, 'id'),
#      State({'type': 'personality-slider', 'index': ALL}, 'id'),
#      State('cytoscape', 'elements')],
#     prevent_initial_call=True
# )
# def update_agent_settings(memory_values, personality_values, memory_ids, personality_ids, elements):
#     # TODO: Re-enable and adapt this callback once backend Agent supports these settings
#     ctx = dash.callback_context
#     if not ctx.triggered:
#         return elements
#
#     # Update memory settings
#     for slider_id, value in zip(memory_ids, memory_values):
#         agent_name = slider_id['index']
#         # backend_agent_manager.agents[agent_name].set_memory_enabled(bool(value)) # Placeholder
#
#     # Update personality settings
#     for slider_id, value in zip(personality_ids, personality_values):
#         agent_name = slider_id['index']
#         # backend_agent_manager.agents[agent_name].set_personality_strength(float(value)) # Placeholder
#
#     return elements  # Return existing elements to refresh display

if __name__ == '__main__':
    app.run(debug=True, port=8050)
