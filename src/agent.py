import json
from src.generation import structured_response
from src.coding import run_python_sandboxed, extract_code
from src.io import read_file, list_files, write_file
from src.ui import status, show_tool_call, show_answer, show_tool_input, show_tool_output, show_thought
from pydantic import BaseModel, TypeAdapter
from typing import Literal

# ---Pydantic Models ---
class ToolCall(BaseModel):
    type: Literal["tool_call"]
    thought: str
    action: Literal["read_file", "list_files", "run_python_sandboxed", "write_file"]
    action_input: dict

class FinalAnswer(BaseModel):
    type: Literal["final_answer"]
    thought: str
    answer: str

AgentResponse = ToolCall | FinalAnswer
adapter = TypeAdapter(AgentResponse)
RESPONSE_SCHEMA = adapter.json_schema()

LLM_MODEL = "qwen3.5:2b"

AVAILABLE_TOOLS = {
    "run_python_sandboxed": run_python_sandboxed,
    "read_file": read_file,
    "list_files": list_files,
    "write_file": write_file,
}

SYSTEM_PROMPT = """
You are an AI agent that runs in a execution loop. You must think step-by-step and decide whether to use a tool or provide your final answer.

Your project directory is: {project_dir}  # The agent can read any file in this directory using the read_file tool, but cannot access files outside of it.
IMPORTANT: if you write a file, make sure to write it to the project directory or a subdirectory of {project_dir}.
You have access to the following tools:
- write_file(path: str, content: str) -> None:
- run_python_sandboxed(code: str) -> str:
- read_file(path: str) -> str:
- list_files(project_dir: str) -> list[str]:

CRITICAL: You MUST reply in EXACTLY one of the following two JSON formats. Do not include any other text, markdown blocks, or commentary.

E.g., If you need to write a file:
{{"type": "tool_call", "thought": "...", "action": "write_file", "action_input": {{"path": "foo.py", "content": "print('hello')"}}}}

If you have the final answer:
{{"type": "final_answer", "thought": "...", "answer": "..."}}
"""

def agent_loop(user_prompt: str, project_dir: str, history: list = None, model: str = LLM_MODEL, max_iter=3):
    system_msg = {"role": "system", "content": SYSTEM_PROMPT.replace("{project_dir}", project_dir)}

    if history:
        # History already contains the system prompt from the first call
        messages = history
    else:
        messages = [system_msg]

    messages.append({"role": "user", "content": user_prompt})

    for i in range(max_iter):
        with status(f"Thinking (step {i+1})"):
            response = structured_response(messages, format=RESPONSE_SCHEMA, model=model)
            response_dict = json.loads(response)

        if response_dict["type"] == "final_answer":
            messages.append({"role": "assistant", "content": response})
            show_thought(response_dict["thought"])
            show_answer(response_dict["answer"])
            return messages

        elif response_dict["type"] == "tool_call":
            tool_name = response_dict["action"]
            tool_input = response_dict["action_input"]

            if tool_name == "run_python_sandboxed":
                tool_input = {"code": extract_code(tool_input["code"])}

            show_thought(response_dict["thought"])
            show_tool_call(tool_name)
            show_tool_input(tool_input)

            with status(f"Running {tool_name}"):
                try:
                    tool_output = AVAILABLE_TOOLS[tool_name](**tool_input)
                except Exception as e:
                    tool_output = f"Error running tool: {str(e)}"

            show_tool_output(tool_output)
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Tool '{tool_name}' returned:\n{tool_output}"})