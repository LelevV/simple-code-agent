# Simple Code Agent

A minimal agentic coding assistant that runs in a ReAct-style loop — it thinks, picks a tool, observes the output, and repeats until it has an answer.

Built with local LLMs via [Ollama](https://ollama.com) and structured output using Pydantic.

## How It Works

The agent follows a **think → act → observe** loop:

1. The LLM receives the conversation history and decides to either call a tool or give a final answer.
2. Tool output is fed back into the conversation as context.
3. The loop repeats until a final answer is produced or `max_iter` is reached.

Responses are constrained to a strict JSON schema (via Pydantic) so the agent always returns a valid tool call or final answer.

## Available Tools

| Tool | Description |
|---|---|
| `read_file(path)` | Read a file from the project directory |
| `list_files(project_dir)` | List all files in the project directory |
| `write_file(path, content)` | Write content to a file |
| `run_python_sandboxed(code)` | Execute Python code in a sandbox |

## Project Structure

```
├── main.py              # CLI entry point
├── src/
│   ├── agent.py         # Agent loop and tool dispatch
│   ├── generation.py    # LLM structured response wrapper
│   ├── coding.py        # Sandboxed Python execution
│   ├── io.py            # File read/write/list utilities
│   └── ui.py            # CLI display helpers
└── project/             # Working directory the agent operates on
```

## Usage

```bash
python main.py
```

Then type natural language commands:

```
You: create a python script that sorts a list of numbers
  ✓ Thinking (step 1)
  → write_file
  ✓ Running write_file
  ✓ Thinking (step 2)
Answer: Created sort_numbers.py in the project directory.
```

Type `exit` to quit.

## Configuration

In `main.py`:

```python
LLM_MODEL = "qwen2.5-coder:7b"  # Any Ollama model
PROJECT_DIR = "./project"         # Agent's working directory
```

## Requirements

- Python 3.10+
- Ollama running locally with your chosen model pulled