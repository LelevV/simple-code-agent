import json

from src.generation import structured_response
from src.coding import run_python_sandboxed, extract_code


from pydantic import BaseModel, TypeAdapter
from typing import Literal

class ToolCall(BaseModel):
    type: Literal["tool_call"]
    thought: str
    action: Literal["run_python"]
    action_input: str


class FinalAnswer(BaseModel):
    type: Literal["final_answer"]
    thought: str
    answer: str


# Define a union type for the agent's response, which can be either a tool call or a final answer.
AgentResponse = ToolCall | FinalAnswer
adapter = TypeAdapter(AgentResponse)
RESPONSE_SCHEMA = adapter.json_schema()

LLM_MODEL = "qwen3.5:2b"

AVAILABLE_TOOLS = {
    "run_python": run_python_sandboxed,
}

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
    for i in range(max_iter):

        print(f"\n--- Iteration {i+1} ---")
        # Step 1: Get the agent's response from the LLM.
        response = structured_response(messages, format=RESPONSE_SCHEMA, model=LLM_MODEL)

        print("Raw LLM response:")
        print(response)

        # Step 2: Parse the response as JSON.
        try:
            response_json = json.loads(response)
        except json.JSONDecodeError:
            print("Failed to parse LLM response as JSON. Please ensure the LLM follows the specified format.")
            break
        
        # Step 3: Check if the agent wants to use a tool or provide a final answer.
        if response_json["type"] == "final_answer":
            print("Agent has provided a final answer:")
            print(response_json["answer"])
            break
        elif response_json["type"] == "tool_call":
            tool_name = response_json["action"]
            tool_input = response_json["action_input"]
            print(f"Agent decided to use tool: {tool_name} with input: {tool_input}")
            tool_func = AVAILABLE_TOOLS[tool_name]
            tool_output = tool_func(tool_input)
            print(f"Tool output: {tool_output}")

            # Add the agent's thought and the tool output to the messages for the next iteration.
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "system", "content": f"Tool output: {tool_output}"})

    
if __name__ == "__main__":
    user_prompt = """
    Write a Python function that returns the sum of squares of the first 110 natural numbers, 
    and then execute it to get the result.
    """

    agent_loop(user_prompt) 
    



