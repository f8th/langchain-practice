from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ('system','You are a helpful custome support agent.'),
    MessagesPlaceholder(variable_name='history'),
    ('human', '{query}')
])

#load history from database
history = []

with open('history.txt', 'r') as f:
    history.extend(f.readlines())

# print(history)

prompt = chat_template.invoke({
    'history': history,
    'query': 'What is the status of my order?'
})
print(prompt)