import openai

client = openai.OpenAI(
  api_key="",  # 换成你在 AiHubMix 生成的密钥
  base_url="https://aihubmix.com/v1"
)

response = client.chat.completions.create(
  model="gemini-3-flash-preview-free",
  messages=[
      {"role": "user", "content": "生命的意义是什么？"}
  ]
)

print(response.choices[0].message.content) # 该模型默认开启思考模式