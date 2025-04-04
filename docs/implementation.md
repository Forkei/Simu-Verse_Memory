# Simu-Verse Agent System Implementation

## Overview
This document outlines the implementation of an agent system for Simu-Verse, a simulation environment where AI agents interact in a 3D space. The backend system manages agent behaviors, memory, and interactions through a sophisticated architecture.

## Core Components

### 1. Agent System
- **Agent Manager**: Coordinates multiple agents, their creation, and lifecycle
- **Agent Class**: Base class for all agents with common functionality
- **Agent Profiles**: Configuration files (.txt) containing system prompts for each agent
- **Tools System**: JSON-based configuration for agent capabilities

### 2. Memory System
- **Memory Creation**: Agents create memories from interactions
- **Memory Storage**: Weaviate vector database with collections per agent
- **Memory Retrieval**: Context-aware memory retrieval for agent decision making
- **Subconscious Agent**: Manages memory operations for each main agent

### 3. Communication System
- **Agent-to-Agent Communication**: Protocol for inter-agent messaging
- **Environment Integration**: Future connection to Unity 3D environment

## Agent Architecture

### Main Agent
Each agent has:
- Custom system prompt (stored as .txt file)
- Access to specific tools (defined in tools.json)
- Memory context (retrieved by subconscious)
- XML-based response format with reflection and tool use

### Subconscious Agent
Each main agent has a paired subconscious agent that:
- Summarizes conversations into memories
- Creates memory queries to retrieve relevant context
- Manages memory importance and categorization
- Provides memory context to the main agent

## Memory Structure
Each memory contains:
- Summary: Concise description of the memory
- Category: Classification from predefined list
- Keywords: Relevant terms for retrieval
- Critical Information: Details to aid decision making
- Importance: Numerical value indicating significance
- Timestamp: Automatically added by the system

## Agent Response Format
Agents respond in XML format with two main sections:

### Reflection Section
```xml
<current_task>What you're currently working on</current_task>
<description>Detailed description of what you're doing and why</description>
<next_steps>What you plan to do next</next_steps>
<goal>Your overall goal for this interaction</goal>
<other_info>Any other relevant information or context</other_info>
```

### Tool Use Section
```xml
<tool_use>
  <tool_name>name_of_tool_to_use</tool_name>
  <parameter name="param1">value1</parameter>
  <parameter name="param2">value2</parameter>
</tool_use>
```

## Available Tools
Agents can use various tools to interact with the environment:
- Movement: Navigate to agents, landmarks, items
- Scanning: Detect nearby agents and objects
- Communication: Talk to other agents
- Thinking: Internal planning and reflection
- Interaction: Manipulate objects (doors, chairs, etc.)
- Do Nothing: Pause action

## Implementation Files

### Agent System
- `agent_manager.py`: Manages agent lifecycle and coordination
- `agent.py`: Base agent class with core functionality
- `agent_factory.py`: Creates agents from configuration
- `tools_manager.py`: Loads and manages available tools

### Memory System
- `memory_manager.py`: Coordinates memory operations
- `subconscious_agent.py`: Implements the subconscious functionality
- `memory_utils.py`: Helper functions for memory operations
- `weaviate_client.py`: Interface to Weaviate database

### Configuration
- `agents/`: Directory containing agent system prompts (.txt files)
- `tools.json`: Configuration for available tools
- `memory_categories.json`: Predefined memory categories

## Implementation Plan
1. Set up basic agent and memory architecture
2. Implement tool system and XML response parsing
3. Create subconscious agent functionality
4. Integrate with Weaviate for memory storage
5. Develop agent-to-agent communication
6. Test with simulated interactions
7. Prepare for Unity 3D integration

## Future Extensions
- Emotional modeling for agents
- Learning and adaptation over time
- More sophisticated memory retrieval algorithms
- Advanced social dynamics between agents
