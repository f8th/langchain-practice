from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='/Users/apple/Desktop/prep/Langchain_Document_Loader/dir',
    glob='**/*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.lazy_load()
 
for doc in docs:
    print(doc.metadata)