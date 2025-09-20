from rag.module.VectorConverter import FileToChromaIndex
from rag.module.ModelConnector import OllamaConnector

class RAGOllama:
    def __init__(self, ollama_base_url="http://localhost:11434", chroma_dir="chroma_store", model="llama3"):
        """
        Initialize the RAG pipeline with Ollama and ChromaDB.
        """
        self.chroma_dir = chroma_dir
        self.file_indexer = FileToChromaIndex(persist_dir=self.chroma_dir)
        self.ollama = OllamaConnector(base_url=ollama_base_url)
        self.model = model
        self.current_kb = None
        self.chat_history = [] 

    def build_index(self, file_path, knowledge_base_name=None):
        if not knowledge_base_name:
            import os
            knowledge_base_name = os.path.splitext(os.path.basename(file_path))[0]

        
        existing_kbs = self.file_indexer.list_knowledge_bases()
        if knowledge_base_name in existing_kbs:
            print(f"Adding file to existing KB: {knowledge_base_name}")
            self.file_indexer.add_to_knowledge_base(file_path, knowledge_base_name)
        else:
            print(f"Creating new KB: {knowledge_base_name}")
            self.file_indexer.create_knowledge_base(file_path, knowledge_base_name)

        self.current_kb = knowledge_base_name
        return knowledge_base_name

    def load_index(self, knowledge_base_name=None):
        """
        Load an existing knowledge base from disk.
        
        Args:
            knowledge_base_name (str): Name of the knowledge base to load.
                                     If None, lists available knowledge bases.
        """
        if knowledge_base_name is None:
            
            available_kbs = self.file_indexer.list_knowledge_bases()
            if not available_kbs:
                raise RuntimeError("No knowledge bases found. Create one first with build_index().")
            
            print("Available knowledge bases:")
            for kb in available_kbs:
                print(f"  - {kb}")
            
            
            knowledge_base_name = available_kbs[0]
            print(f"Loading first available: {knowledge_base_name}")
        
        self.file_indexer.load_knowledge_base(knowledge_base_name)
        self.current_kb = knowledge_base_name
        status, msg = self.file_indexer.is_thoroughly_loaded()
        print(msg)
        return knowledge_base_name

    def switch_knowledge_base(self, knowledge_base_name):
        """
        Switch to a different knowledge base.
        
        Args:
            knowledge_base_name (str): Name of the knowledge base to switch to
        """
        self.file_indexer.load_knowledge_base(knowledge_base_name)
        self.current_kb = knowledge_base_name
        print(f"Switched to knowledge base: {knowledge_base_name}")

    def list_knowledge_bases(self):
        """
        List all available knowledge bases.
        """
        return self.file_indexer.list_knowledge_bases()

    def delete_knowledge_base(self, knowledge_base_name):
        """
        Delete a knowledge base.
        
        Args:
            knowledge_base_name (str): Name of the knowledge base to delete
        """
        self.file_indexer.delete_knowledge_base(knowledge_base_name)
        
        
        if self.current_kb == knowledge_base_name:
            self.current_kb = None

    def query_with_context(self, query_text, top_k=3):
        if not self.current_kb:
            raise RuntimeError("No knowledge base currently selected.")

        
        retriever = self.file_indexer.index.as_retriever(similarity_top_k=top_k)
        relevant_nodes = retriever.retrieve(query_text)
        context = "\n\n".join([node.get_content() for node in relevant_nodes])

        
        history_str = ""
        for role, content in self.chat_history[-5:]:  
            history_str += f"{role}: {content}\n"

        
        augmented_prompt = f"""You are a helpful assistant.
        Use the following conversation history and knowledge base context to answer.

        Conversation so far:
        {history_str}

        Knowledge Base Context:
        {context}

        User: {query_text}
        Assistant:"""

        response = self.ollama.generate(model=self.model, prompt=augmented_prompt)

        
        self.chat_history.append(("User", query_text))
        self.chat_history.append(("Assistant", response.strip()))

        return response

    def get_status(self):
        """
        Get current status of the RAG system.
        """
        status = {
            "model": self.model,
            "chroma_dir": self.chroma_dir,
            "current_kb": self.current_kb,
            "available_kbs": self.file_indexer.list_knowledge_bases()
        }
        
        if self.current_kb and self.file_indexer.current_collection:
            status["current_kb_docs"] = self.file_indexer.current_collection.count()
        
        return status

    def show_status(self):
        """
        Display current status in a readable format.
        """
        print(f"\n=== RAG System Status ===")
        print(f"Model: {self.model}")
        print(f"ChromaDB Directory: {self.chroma_dir}")
        print(f"Current Knowledge Base: {self.current_kb or 'None'}")
        
        if self.current_kb and self.file_indexer.current_collection:
            doc_count = self.file_indexer.current_collection.count()
            print(f"Documents in Current KB: {doc_count}")
        
        print("\nAvailable Knowledge Bases:")
        available = self.file_indexer.list_knowledge_bases()
        if not available:
            print("  No knowledge bases found.")
        else:
            for kb in available:
                marker = " (current)" if kb == self.current_kb else ""
                print(f"  - {kb}{marker}")