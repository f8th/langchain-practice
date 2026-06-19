from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableBranch, RunnableLambda, RunnableSequence
from dotenv import load_dotenv

load_dotenv()

getJokePrompt = PromptTemplate(
    template='Give me a joke about {topic} in Hinglish.',
    input_variables=['topic']
)

explainJokePrompt = PromptTemplate(
    template='Explain me the given joke -> {joke} in Hinglish',
    input_variables=['joke']
)

llm = ChatOpenAI()
parser = StrOutputParser()

# Step 1: topic -> joke (a plain string)
joke_chain = getJokePrompt | llm | parser

# Step 2: given a joke string, fan out into explanation + the joke itself.
#   Input to this parallel block is the joke STRING.
#   - 'explanation': must wrap the string into {'joke': ...} for the prompt
#   - 'joke': RunnablePassthrough() returns the input string unchanged

fan_out = RunnableParallel(
    {
        'explanation': (lambda j: {'joke': j}) | explainJokePrompt | llm | parser,
        'joke': RunnablePassthrough()
    }
)

final_chain = joke_chain | fan_out

# result = final_chain.invoke({'topic': 'layoff'})

# print(result['joke'])
# print(result['explanation']) 

#use of RunnableLambda to create a simple chain that returns a joke and bumber of words in that joke

def count_words(joke: str) -> int:
    return len(joke.split())

fan_out2 = RunnableParallel(
    {
        'joke': RunnablePassthrough(),
        'word_count': RunnableLambda(count_words)
    }
)

new_chain = RunnableSequence (
    joke_chain,
    fan_out2
)

result2 = new_chain.invoke({'topic': 'layoff'})
print(result2['joke'])
print(result2['word_count'])