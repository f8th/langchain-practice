from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(
    template="You are a litrature expert and shakespearean scholar. give me a 3 line poetic summary on {topic} in the style of Shakespeare but in Hinglish.",
    input_variables=["topic"]
)

model = ChatOpenAI(model_name="gpt-5.4")

parser = StrOutputParser()


chain = prompt | model | parser

result = chain.invoke({"topic": "Chhota Bheem"})
print(result)


chain.get_graph().print_ascii()