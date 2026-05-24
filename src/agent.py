import json
from src.generation import structured_response
from src.coding import run_python_sandboxed, extract_code
from src.ui import status, show_tool_call, show_answer
from pydantic import BaseModel, TypeAdapter
from typing import Literal


# ---Pydantic Models ---
class ToolCall(BaseModel):
    type: Literal["tool_call"]
    thought: str
    action: Literal["run_python"]
    action_input: str


class FinalAnswer(BaseModel):
    type: Literal["final_answer"]
    thought: str
    answer: str


AgentResponse = ToolCall | FinalAnswer # Union type for agent response, can be either a tool call or a final answer
adapter = TypeAdapter(AgentResponse) # Create a TypeAdapter for the union type to get JSON schema generation
RESPONSE_SCHEMA = adapter.json_schema()

LLM_MODEL = "qwen3.5:2b"
AVAILABLE_TOOLS = {"run_python": run_python_sandboxed}

SYSTEM_PROMPT = """
You are an AI agent that runs in a execution loop. You must think step-by-step and decide whether to use a tool or provide your final answer.
You have access to the following tools: 
- run_python(code: str) -> str: 
CRITICAL: You MUST reply in EXACTLY one of the following two JSON formats. Do not include any other text, markdown blocks, or commentary.
If you need to run code:
{"type": "tool_call", "thought": "...", "action": "run_python", "action_input": "<code>"}
If you have the final answer:
{"type": "final_answer", "thought": "...", "answer": "..."}
"""



def agent_loop(user_prompt: str, history: list = None, model: str=LLM_MODEL, max_iter=3):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    for i in range(max_iter):
        with status(f"Thinking (step {i+1})"):
            response = structured_response(messages, format=RESPONSE_SCHEMA, model=model)

        response_dict = json.loads(response)

        if response_dict["type"] == "final_answer":
            messages.append({"role": "assistant", "content": response})
            show_answer(response_dict["answer"])
            return messages  # Return full message history for potential further interactions or debugging

        elif response_dict["type"] == "tool_call":
            tool_name = response_dict["action"]
            tool_input = extract_code(response_dict["action_input"]) if tool_name == "run_python" else response_dict["action_input"]
            show_tool_call(tool_name)

            with status(f"Running {tool_name}"):
                tool_output = AVAILABLE_TOOLS[tool_name](tool_input)

            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "system", "content": f"Tool output: {tool_output}"})

