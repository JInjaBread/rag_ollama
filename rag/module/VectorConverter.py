import os
import fitz  # PyMuPDF
from llama_index.core import Document, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb


class FileToChromaIndex:
    def __init__(self, persist_dir="chroma_store"):
        """
        Initialize the ChromaDB client and persist directory.
        """
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=persist_dir)

        self.collection = self.client.get_or_create_collection("file_index")

        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

        self.index = None

    def extract_text_from_pdf(self, file_path):
        """
        Extract text from a PDF file.
        """
        text = ""
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text += page.get_text()
        return text

    def extract_text_from_txt(self, file_path):
        """
        Extract text from a TXT file.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def load_file(self, file_path):
        """
        Load a file and return its text content.
        Supports PDF and TXT.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self.extract_text_from_pdf(file_path)
        elif ext == ".txt":
            return self.extract_text_from_txt(file_path)
        else:
            raise ValueError("Unsupported file format. Use PDF or TXT.")

    def create_index(self, file_path):
        """
        Create a LlamaIndex index from a file and store it in ChromaDB.
        """
        text_content = self.load_file(file_path)
        documents = [Document(text=text_content)]

        self.index = VectorStoreIndex.from_documents(
            documents,
            vector_store=self.vector_store,
            embed_model=self.embed_model
        )
        return self.index

    def query(self, query_text):
        """
        Query the index for relevant results.
        """
        if not self.index:
            raise RuntimeError("Index not created yet. Call create_index() first.")
        query_engine = self.index.as_query_engine()
        return query_engine.query(query_text)
