import json

from src.generation import call_llm
from src.coding import run_python_sandboxed, extract_code


AVAILABLE_TOOLS = {
    "run_python": run_python_sandboxed,
}

SYSTEM_PROMPT = """

You are an AI agent that runs in a execution loop. You must think step-by-step and decide whether to use a tool or provide your final answer.

You have access to the following tools: 
- run_python(code: str) -> str: 

CRITICAL: You MUST reply in EXACTLY one of the following two JSON formats. Do not include any other text, markdown blocks, or commentary.

If you need to use a tool:
{"thought": "Your reasoning here...", "action": "run_python", "action_input": "<python code to run>"}

If you have the final answer to the user prompt:
{"thought": "I have the final answer", "final_answer": "The final message to user."}

"""

def agent_loop(user_prompt: str, max_iter=3):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": user_prompt})
    for i in range(max_iter):

        print(f"\n--- Iteration {i+1} ---")
        # Step 1: Get the agent's response from the LLM.
        response = call_llm(messages, model="qwen3.5:2b")

        print("Raw LLM response:")
        print(response)

        # Step 2: Parse the response as JSON.
        try:
            response_json = json.loads(response)
        except json.JSONDecodeError:
            print("Failed to parse LLM response as JSON. Please ensure the LLM follows the specified format.")
            break
        
        
        # Step 3: Check if the agent wants to use a tool or provide a final answer.
        if "final_answer" in response_json:
            print("Agent has provided a final answer:")
            print(response_json["final_answer"])
            break
        elif "action" in response_json and response_json["action"] in AVAILABLE_TOOLS:
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
    user_prompt = "What is the output of running this code? ```print(sum([i**2 for i in range(10)]))```"
    agent_loop(user_prompt) 


