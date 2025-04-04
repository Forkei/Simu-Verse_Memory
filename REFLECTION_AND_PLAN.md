# SimuVerse Code Reflection and Next Steps Plan

This document outlines the current state of the SimuVerse codebase, identifies areas for cleanup and improvement, and proposes a checklist for the next development steps, with a focus on integrating the memory system.

## Reflection on Current State

### Strengths:
1.  **Modular LLM Support:** `python_backend/src/llm/llm_manager.py` provides a good abstraction for multiple LLM providers (Ollama, Anthropic, OpenAI).
2.  **Memory Foundation:** `python_backend/src/memory/` contains `WeaviateClient`, `MockWeaviateClient`, and `MemoryUtils`, providing a solid base for memory operations. `python_backend/src/agents/subconscious_agent.py` defines logic for memory creation and retrieval.
3.  **Agent Structure:** `python_backend/src/agents/agent.py` defines a base agent structure with conversation history and tool placeholders. `python_backend/src/agents/agent_manager.py` manages multiple agents and their interactions.
4.  **Configuration:** Basic configuration exists for tools (`tools.json`) and memory categories (`memory_categories.json`). Agent profiles are templated (`profiles/`, `templates/`).
5.  **UI Prototype:** `SimuVerse-Test-Environment-V1/ui/simulation_grid_test.py` provides a working Dash/Cytoscape UI for visualizing agent interactions and basic controls.
6.  **Movement Logic:** The UI prototype includes logic for agent movement based on conversation rounds and explicit `[MOVE]` commands.

### Weaknesses & Areas for Improvement:
1.  **Code Duplication/Integration:**
    *   There's significant overlap and potential conflict between the agent/LLM handling in `SimuVerse-Test-Environment-V1/modules/framework.py` and the more recent structure in `python_backend/src/`. The `framework.py` seems partially deprecated but is still used in `simulation_grid_test.py` and `test_agents.py`.
    *   `agent_manager.py` in the test environment (`SimuVerse-Test-Environment-V1/src/`) seems like an older version compared to the one in `python_backend/src/agents/`.
2.  **Memory System Integration:** The memory components (`WeaviateClient`, `SubconsciousAgent`) are defined but not integrated into the main simulation loop or UI in `simulation_grid_test.py`.
3.  **Visualization:** `SimuVerse-Test-Environment-V1/src/visualize.py` uses `matplotlib` and seems disconnected from the main Dash UI. The Dash UI itself could be enhanced (see `IMPROVEMENTS.md`).
4.  **Asynchronous Handling:** The `simulation_step_async` in `simulation_grid_test.py` uses threading to run async code, which isn't ideal for Dash and might cause issues. Proper async integration (e.g., `dash-extensions`) or sticking to synchronous operations is needed.
5.  **Configuration:** API keys and model names are hardcoded or loaded directly from `.env` in `simulation_grid_test.py` instead of using a centralized config system.
6.  **Incomplete Features:** Many Python files in `python_backend/src/` are empty, indicating planned but unimplemented features (subconscious memory editing, logging utilities, vector utils, AI logic, websocket communication).
7.  **Testing:** Lack of a formal testing suite (unit, integration). `test_agents.py` seems outdated.
8.  **Error Handling:** Error handling, especially around API calls and file access, could be more robust.
9.  **Project Structure:** The relationship between `python_backend/` and `SimuVerse-Test-Environment-V1/` needs clarification. Imports in the UI code suggest they might be intended to run together, but the structure implies separation.

## Checklist for Next Steps

### I. Cleanup & Refactoring:
*   [ ] **Consolidate Agent Logic:** Decide on a single source of truth for Agent definition and LLM interaction. Integrate the `LLMManager` from `python_backend` into the agent logic used by the simulation (`simulation_grid_test.py`). Deprecate or remove redundant code in `SimuVerse-Test-Environment-V1/modules/framework.py` and `SimuVerse-Test-Environment-V1/src/agent_manager.py`.
*   [ ] **Remove/Update Old Visualization:** Remove the unused `matplotlib`-based `visualize.py` or update it if it serves a different purpose. Focus enhancements on the Dash/Cytoscape UI.
*   [ ] **Clarify Project Structure:** Define how `python_backend` and the test environment UI are meant to interact. Adjust imports accordingly (e.g., make `python_backend` an installable package or adjust `sys.path`).
*   [ ] **Refactor Async:** Either remove the `simulation_step_async` and focus on synchronous simulation steps or implement proper async handling within the Dash framework.
*   [ ] **Centralize Configuration:** Move API keys, model names, prompts, and other configurations out of scripts and into dedicated config files (`config.yaml`?) or rely solely on `.env` loaded centrally.

### II. Memory System Integration:
*   [ ] **Instantiate Managers:** In `simulation_grid_test.py`, instantiate `LLMManager` and `WeaviateClient` (or `MockWeaviateClient`).
*   [ ] **Instantiate AgentManager:** Pass the `LLMManager` and `WeaviateClient` to the `python_backend.src.agents.agent_manager.AgentManager` constructor.
*   [ ] **Agent Creation:** Modify `simulation_grid_test.py` to use the *backend's* `AgentManager.create_agent` method for creating agents. This will automatically set up the `SubconsciousAgent` as well.
*   [ ] **Simulation Loop Integration:**
    *   Modify `simulation_step` in `simulation_grid_test.py`.
    *   Instead of agents directly calling `generate_response`, the loop should call `AgentManager.process_agent_turn` for the relevant agent.
    *   Pass necessary context (like the message from the other agent) to `process_agent_turn`.
*   [ ] **Memory Context in Prompt:** Ensure the `process_agent_turn` correctly retrieves memories via the `SubconsciousAgent` and injects the `MEMORY_CONTEXT` into the agent's system prompt before calling the LLM.
*   [ ] **Memory Creation:** Ensure `process_agent_turn` triggers memory creation in the `SubconsciousAgent` after a response is generated.
*   [ ] **UI Updates:** Add UI elements (optional) to display retrieved memories or agent reflections derived from memory.
*   [ ] **Configuration:** Ensure Weaviate connection details are configurable (use `.env` or a config file).
*   [ ] **Testing:** Add specific tests for the memory creation/retrieval cycle within the simulation.

### III. Bug Fixes:
*   [ ] **Fix Imports:** Ensure all imports work correctly regardless of how the simulation script is run.
*   [ ] **Address Async Issues:** Implement a stable approach for handling potentially long-running LLM calls within the Dash UI (synchronous with loading states, or proper async integration).

### IV. Feature Implementation & Improvements:
*   [ ] **Implement Tool Usage:** Parse the `<tool_use>` section of agent responses and execute the corresponding actions (starting with `movement`).
*   [ ] **Implement Environment Interaction:** Flesh out tools like `scan` and `interact` by adding a simple environment state representation.
*   [ ] **Enhance UI:** Implement features from `IMPROVEMENTS.md` (e.g., better conversation view, agent detail panels, relationship visualization).
*   [ ] **Implement Personality:** Integrate the personality system fully.
*   [ ] **Logging:** Implement centralized logging using `python_backend/src/utils/logging.py` (once implemented).
*   [ ] **Testing:** Create unit and integration tests.

### V. Documentation:
*   [ ] **Update README:** Reflect the integrated architecture and memory system.
*   [ ] **Add Architecture Docs:** Create `docs/architecture.md` explaining the final project structure.
*   [ ] **Document Memory System:** Explain how memory is created, stored, and retrieved in `docs/implementation.md` or a dedicated memory doc.
