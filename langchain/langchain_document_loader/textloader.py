from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv, parser

load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()

loader = TextLoader('/Users/apple/Desktop/prep/Langchain_Document_Loader/ayush.txt', encoding='utf-8')

documents = loader.load()

# print(documents[0].page_content)

prompt = PromptTemplate(
    template='Summarize about the given text: \n {text}',
    input_variables=['text']
)

chain = prompt | model | parser

result = chain.invoke({'text':  documents[0].page_content})

print(result)