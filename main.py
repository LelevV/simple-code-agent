from src.generation import call_llm
from src.coding import run_python_sandboxed, extract_code


if __name__ == "__main__":
    # Step 1: Generate some code with the LLM.
    messages = [
        {"role": "system", "content": """
         You are a code-only assistant. Output ONLY valid Python code. 
         No explanations, no markdown fences, no comments unless part of the code logic. 
         Your entire response must be directly executable by a Python interpreter.
         """},
        {"role": "user", "content": "Write a Python script that prints 'Hello, World!'."},
    ]
    code = call_llm(messages, model="qwen3.5:2b")  
    print("Raw LLM response:")
    print(code)
    code = extract_code(code)  # In case the LLM included markdown fences or extra text.
    print("Clean code:")
    print(code)

    # Step 2: Run the generated code in a sandbox and capture the output.
    result = run_python_sandboxed(code, timeout=5, max_output_bytes=1000)
    print("\nExecution result:")
    print(result)


