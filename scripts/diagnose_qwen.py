"""Diagnose Qwen3 thinking behaviour through Ollama."""

import ollama

question = "How do convolutional neural networks help detect diabetic retinopathy?"

response = ollama.chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "system",
            "content": ("Answer concisely. Return only the final answer."),
        },
        {
            "role": "user",
            "content": f"/no_think\n\n{question}",
        },
    ],
    think=False,
    options={
        "temperature": 0.1,
        "num_predict": 512,
    },
)

print("\n=== RESPONSE METADATA ===")
print("Done reason:", response.done_reason)
print("Thinking:", repr(response.message.thinking))

print("\n=== CONTENT ===")
print(repr(response.message.content))
