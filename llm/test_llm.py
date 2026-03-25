from dotenv import load_dotenv
load_dotenv(".env.infineon")

from llm.client import InfineonGPTClient


client = InfineonGPTClient()
print(client.generate("Give me 2 short sentences", k=2))