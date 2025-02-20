# python_backend/src/utils/visualization_app.py
import streamlit as st
import time
from typing import List, Dict

class AgentVisualizer:
    def __init__(self, agents: List[Dict]):
        self.agents = agents
        self.memory_updates = {}

    def update_agent_state(self, agent_id: str, action: str, memory_change: str = None):
        """Update an agent's state and memory."""
        if memory_change:
            self.memory_updates[agent_id] = memory_change
        st.session_state[agent_id] = action

    def visualize(self):
        st.title("Subconscious Memory Simulation - Agent Visualization")

        # Sidebar for agent selection
        st.sidebar.header("Agents")
        selected_agent = st.sidebar.selectbox("Select Agent", [f"Agent {i+1}" for i in range(len(self.agents))])

        # Main display: Agent activity
        st.header(f"Agent {selected_agent[-1]} Activity")
        if selected_agent in st.session_state:
            st.write(f"**Action:** {st.session_state[selected_agent]}")

        # Conversations
        st.subheader("Conversations")
        for agent_id, message in self.agents[0]["conversations"].items():
            st.write(f"**{agent_id}:** {message}")

        # Thinking/Decision-Making
        st.subheader("Thinking")
        st.write("🤔 Agent is reasoning about next action...")

        # Tool Use
        st.subheader("Tool Interactions")
        for agent in self.agents:
            if agent["tool_use"]:
                st.write(f"🛠️ {agent['id']} used {agent['tool_use']}")

        # Subconscious/Memory Processes
        st.subheader("Subconscious Activity")
        for agent_id, update in self.memory_updates.items():
            st.write(f"🧠 {agent_id} - {update}")

        # Memory Importance (simplified)
        st.subheader("Memory Importance")
        for agent in self.agents:
            st.progress(agent["memory_importance"], text=f"{agent['id']} Memory Importance")

        # Real-time update simulation
        if st.button("Simulate Next Step"):
            for agent in self.agents:
                action = f"Moving to {agent['location']}" if agent["location"] else "Idle"
                memory_change = "Queried 'door locked' → 'Use key'" if agent["id"] == selected_agent else None
                self.update_agent_state(agent["id"], action, memory_change)
            time.sleep(1)  # Simulate delay
            st.experimental_rerun()

if __name__ == "__main__":
    # Example agents data (simplified for visualization)
    agents = [
        {"id": "Agent 1", "location": "table", "tool_use": "key", "memory_importance": 0.8, "conversations": {"Agent 1": "Need help with door?", "Agent 2": "Coming!"}},
        {"id": "Agent 2", "location": "door", "tool_use": None, "memory_importance": 0.7, "conversations": {"Agent 2": "Coming!", "Agent 1": "Need help with door?"}}
    ]
    visualizer = AgentVisualizer(agents)
    visualizer.visualize()