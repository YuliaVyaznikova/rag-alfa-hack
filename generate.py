from openai import OpenAI
import os

def gen(prompt):
    api_key = os.getenv("POLZA_API_KEY")
    client = OpenAI(
        base_url="https://api.polza.ai/api/v1",
        api_key=api_key,
    )


    completion = client.chat.completions.create(
        model="deepseek/deepseek-chat-v3.1",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return completion.choices[0].message.content