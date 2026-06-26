import openai

client = openai.OpenAI(
  api_key="",  # Replace with your API key generated on AiHubMix
  base_url="https://aihubmix.com/v1"
)

response = client.chat.completions.create(
  model="gemini-3-flash-preview-free",
  messages=[
      {"role": "user", "content": "Who are you?"}
  ]
)

print(response.choices[0].message.content) # Thinking mode is enabled by default for this model