from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, chain
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(
    template="write a 1 page report on {topic} in Hinglish.",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="from the given report:\n{report}\n give only {number} facts in Hinglish.",
    input_variables=["report", "number"]
)

model = ChatOpenAI(model_name="gpt-5.4")

parser = StrOutputParser()

chain1 = prompt | model | parser

chain = (
    RunnablePassthrough.assign(report=chain1) 
    | prompt2 
    | model 
    | parser
)

result = chain.invoke({"topic": "Chhota Bheem", "number": 2})

print(result)

chain.get_graph().print_ascii()