from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()  # reads ANTHROPIC_API_KEY from .env
client = Anthropic()  # picks the key up from the environment

response = client.messages.create(
    model="claude-haiku-5-5",
    max_tokens=200,
    messages=[
        {
            "role": "user",
            "content": "In one sentence, what is an intent in a conversational AI system?",
        }
    ],
)

print(response.content[0].text)
print("input tokens:", response.usage.input_tokens)
print("output tokens:", response.usage.output_tokens)