from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import hashlib
import chromadb

# Specify  embedding model
embed_model = "mistral:latest"

# Step one, Load in the documents
file_path = "./Knowledge/CompanyPolicies&Guidelines.pdf"

# Step three, embed the text as vector store
# Customizable parameters
user_temperature = .8
user_model = "mistral:latest"
user_gpu_count = 1


ollama_emb = OllamaEmbeddings(
    # Specify the locally ran model
    model = user_model,
    temperature= user_temperature,
    num_gpu = user_gpu_count

)

# Function to compute the hash for each file - to be used as the document id
def compute_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

doc_hash = compute_hash(file_path)

# Initialize a light db for vector store and check if the document already exists
chroma_client = chromadb.Client()
knowledge_base = chroma_client.get_or_create_collection(name = "my_collection") # Get OR Create, so it will work even if it already exists

# Add Documents to the collection (knowledge_base)
# knowledge_base.add(
#     documents=[doc_hash],
#     ids=["id 1"]
    
# )




# chroma_db.aadd_documents([doc_hash])
if doc_hash in knowledge_base.peek():
    print("Document already processed, skipping embedding.")
else:
    print("Processing...")
    loader = PyPDFLoader(file_path)

    docs = []

    docs = loader.load()

    print(doc_hash)
    #print(docs)    

    #Step two, Split the documents
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    all_splits = text_splitter.split_documents(docs)

    knowledge_base.add(
    documents=[doc_hash],
    ids=["id 1"]
    
    )

    print(f"Split blog post into {len(all_splits)} sub-documents.")

    # document_ids = chroma_client.add_documents(documents=all_splits)
    print({f"Docuements successfully embedded "})





