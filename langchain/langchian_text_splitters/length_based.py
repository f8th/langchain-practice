from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    file_path='/Users/apple/Desktop/prep/langchain_document_loader/dir/Apple Reader App Compliance Guide.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
    separator=''
)

result = splitter.split_documents(docs)

print(result[0])
