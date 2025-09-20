
# RAG Chat with Ollama & ChromaDB - Web Interface

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using:
- **Ollama** as the LLM for answering questions
- **ChromaDB** for vector storage and retrieval
- **LlamaIndex** for creating and managing document indexes
- **HuggingFace Embeddings** for semantic search
- **Flask Web Interface** for easy interaction

It allows users to **chat with a knowledge base** through both CLI and web interface using local Ollama models.

---

## **Features**
✔ **Web Interface** - Modern, responsive chat interface  
✔ **Multiple Models** - Support for Llama 3, Mistral, Code Llama, etc.  
✔ **Custom Knowledge Bases** - Upload and process your own documents  
✔ **Real-time Chat** - Interactive conversations with typing indicators  
✔ **Local Processing** - Everything runs on your machine  
✔ **Drag & Drop Upload** - Easy document management  
✔ **PDF/TXT/DOC Support** - Multiple file format support  

---

## **Project Structure**
```
rag_ollama/
├── rag/
│   ├── module/
│   │   ├── VectorConverter.py   # Handles file parsing and ChromaDB indexing
│   │   ├── ModelConnector.py    # Handles API calls to Ollama
│   │   └── RagOllama.py         # Main RAG pipeline logic
│   ├── __main__.py              # Main entry point (CLI + Web server)
│   └── app.py                   # CLI interface functions
├── web/
│   └── index.html               # Web chat interface
├── uploads/                     # Uploaded knowledge base files
├── chroma_store/               # ChromaDB vector storage
└── Pipfile                     # Dependencies
```

---

## **Installation**

### **Prerequisites**
- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running locally
- Recommended Ollama models:
  ```bash
  ollama pull llama3
  ollama pull mistral
  ollama pull codellama
  ```

### **Install Dependencies**

**Option 1: Using pipenv (recommended)**
```bash
pipenv install
pipenv shell  # Activate virtual environment
```

**Option 2: Using pip**
```bash
pip install -r requirements.txt
```

**Option 3: Manual installation**
```bash
# Core dependencies
pip install llama-index chromadb llama-index-vector-stores-chroma pymupdf sentence-transformers llama-index-embeddings-huggingface

# Web interface dependencies (optional for CLI-only usage)
pip install flask flask-cors werkzeug requests

# Fix compatibility issues
pip install "numpy>=1.24.0,<2.0.0" "torch>=2.0.0"
```

---

## **Usage**

### **Web Interface (Recommended)**

1. **Start Ollama:**
   ```bash
   ollama serve
   ```

2. **Start the web server:**
   ```bash
   python -m rag --mode web
   ```
   
   Optional parameters:
   ```bash
   python -m rag --mode web --port 8080 --debug
   ```

3. **Open your browser:**
   Navigate to `http://localhost:5000` (or your custom port)

4. **Configure and chat:**
   - Select an Ollama model (Llama 3, Mistral, etc.)
   - Create custom knowledge bases by uploading documents
   - Start chatting with your documents!

### **CLI Interface**

For the original CLI experience (default mode):
```bash
python -m rag
# or explicitly
python -m rag --mode cli
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