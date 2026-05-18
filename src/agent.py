import json
from src.generation import structured_response
from src.coding import run_python_sandboxed, extract_code
from pydantic import BaseModel, TypeAdapter
from typing import Literal


# --- ANSI Colors ---
class C:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    RESET = "\033[0m"


def header(text, color=C.CYAN):
    width = 60
    print(f"\n{color}{C.BOLD}{'─' * width}")
    print(f"  {text}")
    print(f"{'─' * width}{C.RESET}")


def label(name, value="", color=C.YELLOW):
    print(f"  {color}{C.BOLD}{name}{C.RESET} {value}")


def indent(text, prefix="  │ "):
    for line in str(text).splitlines():
        print(f"{C.DIM}{prefix}{C.RESET}{line}")


# --- Models ---
class ToolCall(BaseModel):
    type: Literal["tool_call"]
    thought: str
    action: Literal["run_python"]
    action_input: str


class FinalAnswer(BaseModel):
    type: Literal["final_answer"]
    thought: str
    answer: str


AgentResponse = ToolCall | FinalAnswer
adapter = TypeAdapter(AgentResponse)
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


def agent_loop(user_prompt: str, max_iter=3):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": user_prompt})

    header("AGENT LOOP START", C.MAGENTA)
    label("Model:", LLM_MODEL)
    label("Max iterations:", str(max_iter))
    label("Prompt:")
    indent(user_prompt.strip())

    for i in range(max_iter):
        header(f"Iteration {i + 1}/{max_iter}")

        # Step 1: Get response
        label("Calling LLM...")
        response = structured_response(messages, format=RESPONSE_SCHEMA, model=LLM_MODEL)

        # Compact preview for raw response
        preview = response[:80] + "..." if len(response) > 80 else response
        label("Raw:", f"{C.DIM}{preview}{C.RESET}", color=C.CYAN)

        # Step 2: Parse
        try:
            response_json = json.loads(response)
        except json.JSONDecodeError:
            print(f"\n  {C.RED}{C.BOLD}✗ Failed to parse JSON{C.RESET}")
            label("Full response:")
            try:
                indent(json.dumps(json.loads(response), indent=2))
            except json.JSONDecodeError:
                indent(response)
            break

        # Step 3: Show thought
        thought = response_json.get("thought", "")
        if thought:
            label("Thought:", color=C.CYAN)
            indent(thought)

        # Step 4: Route
        if response_json["type"] == "final_answer":
            header("FINAL ANSWER", C.GREEN)
            indent(response_json["answer"])
            break

        elif response_json["type"] == "tool_call":
            tool_name = response_json["action"]
            tool_input = response_json["action_input"]

            label(f"Tool: {tool_name}", color=C.YELLOW)
            label("Code:", color=C.YELLOW)
            indent(tool_input)

            tool_func = AVAILABLE_TOOLS[tool_name]
            tool_output = tool_func(tool_input)

            label("Output:", color=C.GREEN)
            indent(tool_output)

            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "system", "content": f"Tool output: {tool_output}"})

    else:
        print(f"\n  {C.RED}{C.BOLD}⚠ Max iterations reached without final answer{C.RESET}")