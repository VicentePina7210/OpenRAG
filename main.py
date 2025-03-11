from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# Specify the model and embedding model
embed_model = "mistral:latest"
local_model = "mistral:latest"



# Step one, Load in the documents

file_path = "./Knowledge/NetworkingTerms.pdf"

loader = PyPDFLoader(file_path)

docs = []

docs = loader.load()

print(docs)

#Step two, Split the documents

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
all_splits = text_splitter.split_documents(docs)


print(f"Split blog post into {len(all_splits)} sub-documents.")
# Step three, embed the text as vector store

ollama_emb = OllamaEmbeddings(
    # Specify the locally ran model
    model = local_model
)

vector_store = InMemoryVectorStore(ollama_emb)
document_ids = vector_store.add_documents(documents=all_splits)

print(document_ids[:3])

