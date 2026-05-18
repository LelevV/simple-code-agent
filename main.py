from src.agent import agent_loop


if __name__ == "__main__":
    user_prompt = """
    Debug the following code and fix the error:
    ```python
        def add_numbers(a, b):
            return a + b
        print(add_numbers(2,3,4))
    ```
    """
    agent_loop(user_prompt)