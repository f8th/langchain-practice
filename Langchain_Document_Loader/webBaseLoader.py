import os
os.environ['USER_AGENT'] = 'myapp/1.0'

from langchain_community.document_loaders import WebBaseLoader

url = 'https://raw.githubusercontent.com/ebookapp/ebookapp.github.io/main/photo-recovery.json'

loader = WebBaseLoader(url)

docs = loader.load()

print(docs[0].page_content) 