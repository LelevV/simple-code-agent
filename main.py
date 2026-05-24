from src.agent import agent_loop
from src.ui import get_user_input

LLM_MODEL = "qwen2.5-coder:1.5b"


if __name__ == "__main__":
    # create cli 
    print("Welcome to the Simple Code Agent CLI!")
    print("Ask a question or give a command. The agent will think step-by-step and may run code to arrive at the answer.")
    print("Type 'exit' to quit.\n")
    messages = []  # Keep track of message history for context in the agent loop
    while True:
        user_input = get_user_input()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        messages = agent_loop(user_input, history=messages, model=LLM_MODEL, max_iter=5)
        