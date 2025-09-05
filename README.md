
# RAG Chat with Ollama & ChromaDB

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using:
- **Ollama** as the LLM for answering questions
- **ChromaDB** for vector storage and retrieval
- **LlamaIndex** for creating and managing document indexes
- **HuggingFace Embeddings** for semantic search

It allows users to **chat with a knowledge base** (e.g., company profile PDF) using a local Ollama model.

---

## **Features**
✔ Extracts text from PDF or TXT files  
✔ Builds a vector index using ChromaDB  
✔ Uses HuggingFace embeddings for semantic similarity  
✔ Integrates with Ollama to generate context-aware answers  
✔ Simple CLI-based chat interface  

---

## **Project Structure**
```
rag/
├── module/
│   ├── VectorConverter.py       # Handles file parsing and ChromaDB indexing
│   ├── ModelConnector.py        # Handles API calls to Ollama
│   ├── RagOllama.py             # Main RAG pipeline logic
│
├── app.py                      # CLI interface for chatting with the knowledge base
```

---

## **Installation**

### **Prerequisites**
- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running locally (`ollama serve`)
- Ollama model (e.g., `llama3`) downloaded:  
  ```bash
  ollama pull llama3
  ```

### **Install Dependencies**
```bash
pip install pymupdf chromadb llama-index sentence-transformers requests
```

---

## **Usage**

### **1. Start Ollama**
Make sure Ollama server is running:
```bash
ollama serve
```

### **2. Add Your Knowledge Base**
Place your PDF or TXT file in the project root (e.g., `novatech_company_profile.pdf`).

### **3. Run the Chat**
```bash
python -m rag
```

Example interaction:
```
RAG Chat with NovaTech Knowledge Base (type 'exit' to quit)
You: What does NovaTech specialize in?
```

---

## **Code Overview**

### **1. app.py**
- Starts the chat interface.
- Loads `RAGOllama` instance.
- Builds an index from the knowledge base file.
- Accepts user queries and returns context-based answers.

### **2. VectorConverter.py**
- Extracts text from **PDF** or **TXT**.
- Creates an index using **LlamaIndex** and stores vectors in **ChromaDB**.
- Supports semantic search with **HuggingFace embeddings**.

### **3. RagOllama.py**
- Combines:
  - File indexing (`FileToChromaIndex`)
  - Query retrieval and context injection
  - Ollama model response generation

### **4. ModelConnector.py**
- Handles requests to Ollama API (`/api/generate`).
- Supports both **standard** and **streaming** responses.

---

## **Configuration**
- **Default Model:** `llama3`
- **ChromaDB Directory:** `chroma_store`
- **Ollama Base URL:** `http://localhost:11434`

You can change these in:
- `app.py` → `rag = RAGOllama(model="llama3")`
- `RagOllama.py` constructor

---

## **Example Prompt**
```
You: Give me a summary of NovaTech's services.
```

---