import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from tkinterdnd2 import DND_FILES, TkinterDnD
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from uuid import uuid4
import chromadb
import hashlib
import ollama
import os

# Folder to scan for available documents
PDF_FOLDER = "./Knowledge"

# Setup embedding model and Chroma
user_model = "mistral:latest"
ollama_emb = OllamaEmbeddings(model=user_model, temperature=0.8, num_gpu=1)
persistent_client = chromadb.PersistentClient()
chroma_client = chromadb.Client()
sentence_transformer_ef = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-mpnet-base-v2"
)
knowledge_base = chroma_client.get_or_create_collection(
    name="collection_1", embedding_function=sentence_transformer_ef
)

def compute_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

class QAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📄 Document QA Assistant")
        self.root.geometry("960x720")
        self.root.configure(bg="#1e1e1e")
        self.current_doc = None

        self.text_display = ScrolledText(root, wrap=tk.WORD, font=("Segoe UI", 12), bg="#111", fg="#eee", insertbackground="white")
        self.text_display.pack(fill=tk.BOTH, expand=True, padx=20, pady=(20, 10))
        self.text_display.insert(tk.END, "💡 Drag a PDF here, upload, or select from the dropdown to begin.\n")

        entry_frame = tk.Frame(root, bg="#1e1e1e")
        entry_frame.pack(fill=tk.X, padx=20)

        self.input_entry = tk.Entry(entry_frame, font=("Segoe UI", 12), bg="#2e2e2e", fg="white", insertbackground="white")
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=10)
        self.input_entry.bind("<Return>", self.ask_question)

        tk.Button(entry_frame, text="🔎 Ask", command=self.ask_question, bg="#3a3a3a", fg="white", padx=12, pady=4).pack(side=tk.RIGHT, padx=5)

        control_frame = tk.Frame(root, bg="#1e1e1e")
        control_frame.pack(fill=tk.X, padx=20, pady=(0, 10))

        tk.Button(control_frame, text="📂 Upload PDF", command=self.upload_pdf, bg="#333", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="❌ Unload", command=self.unload_document, bg="#333", fg="white").pack(side=tk.LEFT, padx=5)

        tk.Label(control_frame, text="📑 Select Existing:", bg="#1e1e1e", fg="white").pack(side=tk.LEFT, padx=5)

        self.file_selector = ttk.Combobox(control_frame, width=40, state="readonly")
        self.file_selector.pack(side=tk.LEFT, padx=5)
        self.file_selector.bind("<<ComboboxSelected>>", self.select_existing_file)

        tk.Button(control_frame, text="🚪 Exit", command=self.root.quit, bg="#333", fg="white").pack(side=tk.RIGHT, padx=5)

        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.drop_file)

        self.populate_dropdown()

    def populate_dropdown(self):
        if not os.path.exists(PDF_FOLDER):
            os.makedirs(PDF_FOLDER)
        pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
        self.file_selector["values"] = pdf_files

    def upload_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if file_path:
            self.load_document(file_path)

    def drop_file(self, event):
        file_path = event.data.strip().strip("{}")
        if os.path.isfile(file_path) and file_path.lower().endswith(".pdf"):
            self.load_document(file_path)
        else:
            messagebox.showerror("Invalid file", "Please drop a valid PDF file.")

    def select_existing_file(self, event):
        filename = self.file_selector.get()
        full_path = os.path.join(PDF_FOLDER, filename)
        if os.path.exists(full_path):
            self.load_document(full_path)

    def load_document(self, file_path):
        self.unload_document()
        self.current_doc = file_path
        doc_hash = compute_hash(file_path)

        if doc_hash in knowledge_base.peek():
            self.text_display.insert(tk.END, f"\n📄 Document already embedded. Ready for questions.\n")
            return

        self.text_display.insert(tk.END, "\n📚 Loading and embedding document...\n")
        self.text_display.see(tk.END)
        self.root.update_idletasks()

        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = CharacterTextSplitter(
            separator="\n\n", chunk_size=1000, chunk_overlap=200, length_function=len
        )
        all_splits = text_splitter.split_documents(docs)

        ids = [str(uuid4()) for _ in range(len(all_splits))]
        documents = [doc.page_content for doc in all_splits]
        metadatas = [doc.metadata for doc in all_splits]

        knowledge_base.add(documents=documents, ids=ids, metadatas=metadatas)
        self.text_display.insert(tk.END, f"✅ Embedded {len(all_splits)} segments from '{os.path.basename(file_path)}'. Ask away!\n")
        self.text_display.see(tk.END)

    def unload_document(self):
        self.current_doc = None
        self.text_display.delete(1.0, tk.END)
        self.text_display.insert(tk.END, "📤 Document unloaded. Upload, drop, or select a new one.\n")

    def ask_question(self, event=None):
        if not self.current_doc:
            self.text_display.insert(tk.END, "\n⚠️ No document loaded.\n")
            return

        query = self.input_entry.get().strip()
        if not query:
            return

        self.text_display.insert(tk.END, f"\n🟦 You: {query}\n🕐 Processing...\n")
        self.text_display.see(tk.END)
        self.input_entry.delete(0, tk.END)
        self.root.update_idletasks()

        results = knowledge_base.query(query_texts=[query], n_results=3)
        chunks = results.get("documents", [[]])[0]

        if not chunks:
            self.text_display.insert(tk.END, "❌ No relevant information found.\n")
        else:
            context = "\n\n".join(chunks)
            prompt = f"""Only use the following document excerpts to answer the question, if the question is not related or you do not know, directly say so:

{context}

Question: {query}

Answer:"""

            response = ollama.chat(
                model=user_model,
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )

            answer = response["message"]["content"]
            self.text_display.insert(tk.END, f"🟩 Mistral: {answer}\n")
            self.text_display.see(tk.END)

if __name__ == "__main__":
    app = TkinterDnD.Tk()
    QAApp(app)
    app.mainloop()
