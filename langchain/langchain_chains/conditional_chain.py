from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableBranch
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="The sentiment of the feedback")

model = ChatOpenAI(model_name="gpt-5.4")

parser = StrOutputParser()

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Based on the given feedback, analyze the sentiment of the feedback and classify it as positive or negative. \n {feedback} \n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={
        "format_instructions": parser2.get_format_instructions()
    }
)

posFeedbackPrompt = PromptTemplate(
    template='Write an appropriate response to the positive feedback: \n {feedback}',
    input_variables=['feedback']
)

negFeedbackPrompt = PromptTemplate(
    template='Write an appropriate response to the negative feedback: \n {feedback}',
    input_variables=['feedback']
)
 
classifier_chain = prompt1 | model | parser2

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', posFeedbackPrompt | model | parser),
    (lambda x: x.sentiment == 'negative', negFeedbackPrompt | model | parser),
    RunnableLambda(lambda x: "No response generated for the given feedback.")
)

chain = classifier_chain | branch_chain

result = chain.invoke({"feedback": "I am amazed by the new features of your product! It's really terrible good and amazing has improved my workflow significantly along with the cost."})

# print(result)

chain.get_graph().print_ascii()