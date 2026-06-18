from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate

chat_template = ChatPromptTemplate(
    [
        (
            'system','You are a helpful assistant expert in {domain}.'
        ),
        (
            'human','Hi explain me abour {topic}.'
        )
    ]
)

prompt = chat_template.invoke({
    'domain':'AI',
    'topic':'LangChain'
})

print(prompt)