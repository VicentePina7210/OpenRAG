import os
import hashlib
import uuid
import ollama
import chromadb
from colorama import Fore, Style, init

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from chromadb.utils import embedding_functions

# Initialize colorama
init(autoreset=True)

# Constants
DOC_FOLDER = "./Knowledge"
COLLECTION_NAME = "collection_1"
EMBED_MODEL = "mistral:latest"
TRANSFORMER_MODEL = "all-mpnet-base-v2"
USER_TEMPERATURE = 0.8
USER_GPU_COUNT = 1

# Setup
chroma_client = chromadb.Client()
persistent_client = chromadb.PersistentClient()
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=TRANSFORMER_MODEL
)
knowledge_base = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=sentence_transformer_ef,
)
ollama_emb = OllamaEmbeddings(
    model=EMBED_MODEL,
    temperature=USER_TEMPERATURE,
    num_gpu=USER_GPU_COUNT
)

def compute_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def list_documents():
    files = [f for f in os.listdir(DOC_FOLDER) if f.endswith(".pdf")]
    if not files:
        print(Fore.RED + "No PDF documents found in the folder.")
        return []
    print(Fore.CYAN + "\nAvailable documents:")
    for idx, f in enumerate(files, start=1):
        print(Fore.YELLOW + f"  {idx}. {f}")
    return files

def embed_document(file_path):
    doc_hash = compute_hash(file_path)
    if doc_hash in knowledge_base.peek():
        print(Fore.GREEN + "Document already processed, skipping embedding.")
        return

    print(Fore.BLUE + "Embedding document...")
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    all_splits = text_splitter.split_documents(docs)

    local_vector_store = Chroma(
        client=persistent_client,
        embedding_function=ollama_emb,
        collection_name=COLLECTION_NAME
    )

    ids = [str(uuid.uuid4()) for _ in all_splits]
    documents = [doc.page_content for doc in all_splits]
    metadatas = [doc.metadata for doc in all_splits]

    knowledge_base.add(documents=documents, ids=ids, metadatas=metadatas)
    print(Fore.GREEN + f"Embedded {len(all_splits)} chunks.")

def ask_question(question):
    results = knowledge_base.query(query_texts=[question], n_results=3)
    retrieved_texts = results.get("documents", [[]])[0]

    if not retrieved_texts:
        print(Fore.RED + "No relevant information found.")
        return

    context = "\n\n".join(retrieved_texts)
    prompt = f"""Use the following document excerpts to answer the question:

{context}

Question: {question}

Answer:"""

    print(Fore.BLUE + "\nGenerating response from Mistral...\n")
    response = ollama.chat(
        model=EMBED_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    print(Fore.MAGENTA + "\nModel's Answer:\n")
    print(Fore.WHITE + response["message"]["content"])

def main():
    while True:
        files = list_documents()
        if not files:
            break

        try:
            choice = int(input(Fore.CYAN + "\nSelect a document by number (or 0 to exit): "))
            if choice == 0:
                print(Fore.CYAN + "Goodbye!")
                break
            file_name = files[choice - 1]
        except (ValueError, IndexError):
            print(Fore.RED + "Invalid selection. Please try again.")
            continue

        file_path = os.path.join(DOC_FOLDER, file_name)
        embed_document(file_path)

        while True:
            question = input(Fore.GREEN + "\nAsk a question (or type 'switch' to choose another file, 'exit' to quit): ").strip()
            if question.lower() == 'exit':
                print(Fore.CYAN + "Goodbye!")
                return
            elif question.lower() == 'switch':
                break
            elif question:
                ask_question(question)

if __name__ == "__main__":
    main()
