import os
import fitz  
from llama_index.core import Document, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser
import chromadb

class FileToChromaIndex:
    def __init__(self, persist_dir="chroma_store"):
        """
        Initialize ChromaDB client. All knowledge bases stored as collections in one database.
        """
        self.persist_dir = persist_dir
        
        
        sqlite_path = os.path.join(persist_dir, "chroma.sqlite3")
        if os.path.exists(sqlite_path):
            print(f"Found existing ChromaDB at: {sqlite_path}")
        else:
            print(f"Creating new ChromaDB at: {persist_dir}")
            os.makedirs(persist_dir, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        
        self.current_kb = None
        self.current_collection = None
        self.vector_store = None
        self.index = None

    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF file."""
        text = ""
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text += page.get_text()
        return text

    def extract_text_from_txt(self, file_path):
        """Extract text from TXT file."""
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def load_file(self, file_path):
        """Load file and return text content. Supports PDF and TXT."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self.extract_text_from_pdf(file_path)
        elif ext == ".txt":
            return self.extract_text_from_txt(file_path)
        else:
            raise ValueError("Unsupported file format. Use PDF or TXT.")

    def add_to_knowledge_base(self, file_path, knowledge_base_name):
        """Add a file's content into an existing knowledge base."""
        if not self.current_collection or self.current_kb != knowledge_base_name:
            self.load_knowledge_base(knowledge_base_name)

        
        text_content = self.load_file(file_path)
        document = Document(text=text_content)

        
        parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)
        nodes = parser.get_nodes_from_documents([document])

        
        self.index.insert_nodes(nodes)

        count = self.current_collection.count()
        print(f"Added {len(nodes)} chunks from {file_path} → '{knowledge_base_name}' (now {count} chunks total)")
        return count

    def create_knowledge_base(self, file_path, knowledge_base_name):
        """Create new knowledge base (first file)."""
        print(f"Creating knowledge base '{knowledge_base_name}' from {file_path}...")

        
        self.current_collection = self.client.get_or_create_collection(knowledge_base_name)
        self.vector_store = ChromaVectorStore(chroma_collection=self.current_collection)

        
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model
        )

        self.current_kb = knowledge_base_name

        
        return self.add_to_knowledge_base(file_path, knowledge_base_name)

    def load_knowledge_base(self, knowledge_base_name):
        """
        Load an existing knowledge base.
        
        Args:
            knowledge_base_name (str): Name of the knowledge base to load
        """
        try:
            print(f"Loading knowledge base '{knowledge_base_name}'...")
            
            
            self.current_collection = self.client.get_collection(knowledge_base_name)
            self.vector_store = ChromaVectorStore(chroma_collection=self.current_collection)

            
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                embed_model=self.embed_model
            )
            
            self.current_kb = knowledge_base_name
            doc_count = self.current_collection.count()
            print(f"✅ Knowledge base '{knowledge_base_name}' loaded successfully! ({doc_count} documents)")
            return self.index
            
        except Exception as e:
            raise RuntimeError(f"Failed to load knowledge base '{knowledge_base_name}': {str(e)}")

    def list_knowledge_bases(self):
        """List all available knowledge bases."""
        collections = self.client.list_collections()
        kb_names = [col.name for col in collections]
        
        print("Available Knowledge Bases:")
        if not kb_names:
            print("  No knowledge bases found.")
        else:
            for i, kb in enumerate(kb_names, 1):
                try:
                    col = self.client.get_collection(kb)
                    doc_count = col.count()
                    print(f"  {i}. {kb} ({doc_count} documents)")
                except:
                    print(f"  {i}. {kb} (error reading)")
        
        return kb_names

    def delete_knowledge_base(self, knowledge_base_name):
        """Delete a knowledge base."""
        try:
            self.client.delete_collection(knowledge_base_name)
            print(f"✅ Knowledge base '{knowledge_base_name}' deleted successfully.")
            
            
            if self.current_kb == knowledge_base_name:
                self.current_kb = None
                self.current_collection = None
                self.vector_store = None
                self.index = None
        except Exception as e:
            print(f"❌ Failed to delete knowledge base '{knowledge_base_name}': {str(e)}")

    def query(self, query_text):
        """Query the currently loaded knowledge base."""
        if not self.index:
            raise RuntimeError("No knowledge base loaded. Call load_knowledge_base() or create_knowledge_base() first.")
        
        print(f"🔍 Querying knowledge base '{self.current_kb}': {query_text}")
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query_text)
        return response

    def get_current_info(self):
        """Get info about currently loaded knowledge base."""
        if not self.current_kb:
            return {"status": "No knowledge base loaded"}
        
        doc_count = self.current_collection.count() if self.current_collection else 0
        return {
            "current_kb": self.current_kb,
            "document_count": doc_count,
            "database_path": os.path.join(self.persist_dir, "chroma.sqlite3")
        }

    def show_status(self):
        """Show current status and available knowledge bases."""
        print(f"\n📊 ChromaDB Status")
        print(f"Database: {os.path.join(self.persist_dir, 'chroma.sqlite3')}")
        
        if self.current_kb:
            info = self.get_current_info()
            print(f"Currently Loaded: {info['current_kb']} ({info['document_count']} documents)")
        else:
            print("Currently Loaded: None")
        
        print()
        self.list_knowledge_bases()

    def is_thoroughly_loaded(self):
        if not self.current_collection:
            return False, "No collection loaded"
        
        doc_count = self.current_collection.count()
        if doc_count == 0:
            return False, "Collection is empty"
        
        sample = self.current_collection.peek(limit=1)
        if not sample["documents"]:
            return False, "No documents found in collection"
        
        return True, f"Loaded successfully with {doc_count} documents"
    
