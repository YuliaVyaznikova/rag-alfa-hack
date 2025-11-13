from openai import OpenAI

def gen(prompt):
    # api_key = "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"
    # api_key = "ak_QIigRl1dxntO16M3eWjfuficZTRS10ZTdRIN8syWaao"
    api_key = "ak_OBjTnefLZfJmeSfmis7bCP8WbrE84qXBmeLJONFCveA"
    client = OpenAI(
        # base_url="https://openrouter.ai/api/v1",
        base_url="https://api.polza.ai/api/v1",
        api_key=api_key,
    )


    completion = client.chat.completions.create(
        # model="deepseek/deepseek-chat-v3.1:free",
        # model="x-ai/grok-4-fast:free",
        # model="openai/gpt-oss-20b:free",
        # model="meituan/longcat-flash-chat:free",
        model="deepseek/deepseek-chat-v3.1",
        # model = "deepseek/deepseek-v3.1-terminus",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return completion.choices[0].message.content