from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from uuid import uuid4
import chromadb
from chromadb.utils import embedding_functions
import hashlib
import ollama

# Specify  embedding model
embed_model = "mistral:latest"

# Models for Sentence transformers
light_transformer_model = "all-MiniLM-L6-v2" #80mb 49% avg
heavy_transformer_model = "all-mpnet-base-v2" # 420mb and 63%avg

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

# Sentence transformer
chroma_client = chromadb.Client()
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name = heavy_transformer_model
    )

# Create the collection for querying and storing embeddings
knowledge_base = chroma_client.get_or_create_collection(
    name = "collection_1",
    embedding_function= sentence_transformer_ef
    ) # Get OR Create, so it will work even if it already exists

persistent_client = chromadb.PersistentClient() # Not recomended for production



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

    print(f"Text split successful") # Debugging

    local_vector_store = Chroma(
    client = persistent_client,
    embedding_function = ollama_emb,
    collection_name = "collection_1"
    )

    ids = [str(uuid4()) for _ in range(len(all_splits))]  # Unique IDs
    documents = [doc.page_content for doc in all_splits]  # Extract text
    metadatas = [doc.metadata for doc in all_splits]  # Extract metadata


    knowledge_base.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas
    
    )



    print(f"Split blog post into {len(all_splits)} sub-documents.")

    # document_ids = chroma_client.add_documents(documents=all_splits)
    print({f"Documents successfully embedded "})

user_query_text = input("\n Knowledge base ready, ask a question about the documents: ").strip()


# Query the vector store
results = knowledge_base.query(
    query_texts=[user_query_text],
    n_results=3  # Get top 3 most relevant chunks
)

# Extract the retrieved text chunks
retrieved_texts = results.get("documents", [[]])[0]  # Get list of matching text chunks

if not retrieved_texts:
    print("No relevant information found in the document.")
else:
    # Format retrieved text as context for Mistral
    context = "\n\n".join(retrieved_texts)

    # Create a prompt for the LLM
    prompt = f"""Use the following document excerpts to answer the question:

    {context}

    Question: {user_query_text}

    Answer:"""

    print("\nGenerating response from Mistral...\n")

    # Send prompt to Mistral via Ollama API
    response = ollama.chat(
        model="mistral:latest",
        messages=[{"role": "user", "content": prompt}],
        stream = False
        )

    # Print the generated response
    print("\nModels Anwser:\n")
    print(response["message"]["content"])

"""
Notes for future devlopment:
Allow users to select a file from the knowledge base
Knowledge bases will have multiple collections of documents if needed
Users can connect to other hosted serves of ollama 
Implement Messages from Ollama so that users can ask follow up questions
Allow users to connect to OpenAI api and similar llm providers

To fix:

UI Development
Users can modify prompting in UI, having different presets for models
Users can select models from what is stored on their instance of ollama.
Custom parameter adjustment
Stream response

"""