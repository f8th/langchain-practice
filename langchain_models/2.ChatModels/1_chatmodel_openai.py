from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model_name="gpt-5.4", temperature=2.0, max_tokens=10)

result = model.invoke('How ballia UP looks like? in a cinematic way')
print(result.content)

