# Simu-Verse Agent System

A simulation environment where AI agents interact in a 3D space with memory and context awareness.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/simu-verse.git
cd simu-verse
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a `.env` file from the template:
```bash
cp .env.template .env
```

4. Edit the `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OLLAMA_BASE_URL=http://localhost:11434
MODEL_NAME=qwen2.5:14b
WEAVIATE_URL=http://localhost:8080
```

## Running with Weaviate (Optional)

If you want to use Weaviate for persistent memory storage:

1. Make sure Docker is installed on your system
2. Start Weaviate using Docker Compose:
```bash
docker-compose up -d
```

## Running the Agent System

Run the main script:
```bash
python main.py
```

This will start an interactive session with the agent system. You can:
- Chat with the default agent (Alice)
- Create new agents
- Switch between agents
- Exit the system

## Commands

- `exit` or `quit`: Exit the program
- `help`: Show help message
- `agents`: List all agents
- `create <name>`: Create a new agent
- `switch <name>`: Switch to a different agent

## Project Structure

- `python_backend/src/agents/`: Agent implementation
- `python_backend/src/llm/`: LLM integration
- `python_backend/src/memory/`: Memory system
- `python_backend/src/config/`: Configuration files
- `docs/`: Documentation

## License

[MIT License](LICENSE)
