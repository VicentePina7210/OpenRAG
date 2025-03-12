from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import hashlib
# Specify the model and embedding model
embed_model = "mistral:latest"
local_model = "mistral:latest"

# Step one, Load in the documents
file_path = "./Knowledge/NetworkingTerms.pdf"

# Step three, embed the text as vector store
ollama_emb = OllamaEmbeddings(
    # Specify the locally ran model
    model = local_model
)

# Function to compute the hash for each file - to be used as the document id
def compute_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()
# Initialize a light db for vector store and check if the document already exists
db = Chroma(persist_directory = "./chroma_db", embedding_function = ollama_emb)

stored_ids = db.get(['ids'])
doc_hash = compute_hash(file_path)
if doc_hash in stored_ids:
    print("Document already processed, skipping embedding.")
else:
    print("Processing...")
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

    document_ids = db.add_documents(documents=all_splits)

    db.persist()

    print({f"Docuements successfully embedded "})

print(document_ids[:3])




