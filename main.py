from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
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

# Initialize a light db for vector store

db = Chroma(persist_directory = "./chroma_db", embedding_function = ollama_emb)

document_ids = db.add_documents(documents=all_splits)

db.persist()

print(document_ids[:3])


