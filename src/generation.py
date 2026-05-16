import requests


def call_llm(messages: list[dict], model: str = "qwen3.5:4b") -> str:
    """Call local Ollama model."""
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": model, "messages": messages, "stream": False},
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]




if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    response = call_llm(messages)
    print(response)