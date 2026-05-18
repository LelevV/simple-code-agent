import requests
from ollama import chat
from pydantic import BaseModel


def call_llm(messages: list[dict], model: str = "qwen3.5:4b") -> str:
    """Call local Ollama model."""
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": model, "messages": messages, "stream": False},
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def structured_response(messages: list[dict], format, model: str = "qwen3:8b") -> dict:
    """Call local Ollama model and get structured response, using pydantic model."""

    response = chat(
        model=model,
        messages=messages,
        format=format, # Pass JSON schema here
        options={'temperature': 0}
    )
    return response.message.content


if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    response = call_llm(messages, model="qwen3.5:2b")
    print("Raw LLM response:")
    print(response)

    class Asnwer(BaseModel):
        answer: str
        reason: str
    
    structured_resp = structured_response(messages, format=Asnwer, model="qwen3.5:2b")

    print("\nStructured response:")
    print(structured_resp)