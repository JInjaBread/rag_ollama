import argparse
import logging
import os
import warnings
import time
import json
from flask import Response

# Suppress NumPy/PyTorch compatibility warnings
warnings.filterwarnings("ignore", message="Failed to initialize NumPy")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

try:
    from flask import Flask, request, jsonify, send_from_directory
    from flask_cors import CORS
    from werkzeug.utils import secure_filename
    FLASK_AVAILABLE = True
except ImportError as e:
    FLASK_AVAILABLE = False
    print(f"Warning: Flask dependencies not available: {e}")
    print("Web mode will not be available. Install with: pip install flask flask-cors")

from rag.app import chat
from rag.module.RagOllama import RAGOllama
from rag.module.VectorConverter import FileToChromaIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_web_app():
    """Create and configure Flask web application"""
    if not FLASK_AVAILABLE:
        raise ImportError("Flask dependencies not available. Install with: pip install flask flask-cors")
    
    app = Flask(__name__, static_folder='../web')
    CORS(app)
    
    # Global RAG instance cache
    rag_instances = {}
    UPLOAD_FOLDER = 'uploaded_files'  # Separate folder for uploaded files
    CHROMA_FOLDER = "chroma_store"    # ChromaDB storage
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'md'}
    
    # Create necessary directories
    for folder in [UPLOAD_FOLDER, CHROMA_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    @app.route('/')
    def index():
        """Serve the main chatbot interface"""
        return send_from_directory('../web', 'index.html')
    
    @app.route('/<path:filename>')
    def static_files(filename):
        """Serve static files from web directory"""
        return send_from_directory('../web', filename)
    
    @app.route('/api/models', methods=['GET'])
    def get_available_models():
        """Get list of available Ollama models"""
        try:
            models = [
                {"value": "llama2", "name": "Llama 2"},
                {"value": "llama3", "name": "Llama 3"},
                {"value": "llama3.1", "name": "Llama 3.1"},
                {"value": "mistral", "name": "Mistral"},
                {"value": "codellama", "name": "Code Llama"}
            ]
            return jsonify({"models": models})
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/knowledge-bases', methods=['POST'])
    def create_knowledge_base():
        """Create a new knowledge base from uploaded files"""
        try:
            kb_name = request.form.get('name')
            description = request.form.get('description', '')
            
            if not kb_name:
                return jsonify({"error": "Knowledge base name is required"}), 400
                
            files = request.files.getlist('files')
            if not files or len(files) == 0:
                return jsonify({"error": "At least one file is required"}), 400

            # Create upload directory for this KB
            kb_upload_dir = os.path.join(UPLOAD_FOLDER, kb_name)
            if not os.path.exists(kb_upload_dir):
                os.makedirs(kb_upload_dir)
            
            uploaded_files = []
            processed_files = []
            
            # Save uploaded files
            for file in files:
                if file and file.filename and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(kb_upload_dir, filename)
                    file.save(filepath)
                    uploaded_files.append(filename)
            
            if not uploaded_files:
                return jsonify({"error": "No valid files uploaded"}), 400
            
            # Initialize FileToChromaIndex with shared ChromaDB
            indexer = FileToChromaIndex(persist_dir=CHROMA_FOLDER)
            
            # Process each uploaded file and add to the knowledge base
            for filename in uploaded_files:
                filepath = os.path.join(kb_upload_dir, filename)
                try:
                    # Create knowledge base with the first file, then you'd need to 
                    # modify FileToChromaIndex to support adding multiple files to same KB
                    if not processed_files:  # First file creates the KB
                        indexer.create_knowledge_base(filepath, kb_name)
                    else:
                        # For additional files, you'd need to implement add_to_knowledge_base
                        # For now, we'll just create one KB per file upload batch
                        pass
                    processed_files.append(filename)
                except Exception as e:
                    logger.error(f"Error processing file {filename}: {e}")
                    continue
            
            if not processed_files:
                return jsonify({"error": "No files were successfully processed"}), 400
            
            return jsonify({
                "message": f"Knowledge base '{kb_name}' created successfully",
                "files": processed_files,
                "description": description
            })
            
        except Exception as e:
            logger.error(f"Error creating knowledge base: {e}")
            return jsonify({"error": str(e)}), 500
        
    @app.route('/api/knowledge-bases', methods=['GET'])
    def list_knowledge_bases():
        """List all knowledge bases stored in ChromaDB"""
        try:
            # Initialize indexer to access ChromaDB
            indexer = FileToChromaIndex(persist_dir=CHROMA_FOLDER)
            
            # Get all available knowledge bases
            kb_list = indexer.list_knowledge_bases()
            
            logger.info(f"Found knowledge bases: {kb_list}")
            return jsonify({"knowledgeBases": kb_list})
            
        except Exception as e:
            logger.error(f"Error listing knowledge bases: {e}")
            return jsonify({"error": str(e), "knowledgeBases": []})
    
    @app.route('/api/chat', methods=['POST'])
    def web_chat():
        """Handle chat requests with RAG"""
        try:
            data = request.json
            message = data.get('message', '')
            model = data.get('model', 'llama3')
            knowledge_base = data.get('knowledgeBase', '')
            temperature = float(data.get('temperature', 0.7))
            stream = data.get('stream', False)

            if not message:
                return jsonify({"error": "Message is required"}), 400

            if not knowledge_base:
                return jsonify({"error": "Knowledge base is required"}), 400

            # Create unique cache key for RAG instances
            rag_key = f"{model}_{knowledge_base}"
            logger.info(f"Processing chat request - Model: {model}, KB: {knowledge_base}")
            
            # Check if we have cached RAG instance
            if rag_key not in rag_instances:
                logger.info(f"Creating new RAG instance for {rag_key}")
                
                # Create RAG instance
                rag = RAGOllama(model=model, chroma_dir=CHROMA_FOLDER)

                # Try to load the specified knowledge base
                try:
                    rag.load_index(knowledge_base)
                    logger.info(f"Successfully loaded knowledge base: {knowledge_base}")
                except Exception as e:
                    logger.error(f"Failed to load knowledge base '{knowledge_base}': {e}")
                    
                    # Fallback to sample PDF if available
                    sample_pdf = '../novatech_company_profile.pdf'
                    if os.path.exists(sample_pdf):
                        logger.info("Using fallback sample PDF")
                        rag.build_index(sample_pdf, "fallback_kb")
                    else:
                        return jsonify({"error": f"Knowledge base '{knowledge_base}' not found and no fallback available"}), 404

                rag_instances[rag_key] = rag
            else:
                logger.info(f"Using cached RAG instance for {rag_key}")

            rag = rag_instances[rag_key]
            
            # Handle streaming or regular response
            if stream:
                return Response(
                    stream_response(rag, message),
                    mimetype='text/event-stream',
                    headers={
                        'Cache-Control': 'no-cache',
                        'Connection': 'keep-alive',
                        'Access-Control-Allow-Origin': '*'
                    }
                )
            else:
                # Regular response
                response = rag.query_with_context(message)
                return jsonify({"response": response})

        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return jsonify({"error": str(e)}), 500

    def stream_response(rag, message):
        """Generate streaming response by breaking text into chunks"""
        try:
            # Get the full response
            full_response = rag.query_with_context(message)
            
            # Stream the response word by word
            words = full_response.split(' ')
            
            for i, word in enumerate(words):
                # Send each word with a space (except for the last word)
                chunk = word + (' ' if i < len(words) - 1 else '')
                
                # Format as Server-Sent Events
                data = json.dumps({"response": chunk})
                yield f"data: {data}\n\n"
                
                # Small delay to simulate typing
                time.sleep(0.05)  # 50ms delay between words
                
        except Exception as e:
            error_data = json.dumps({"error": str(e)})
            yield f"data: {error_data}\n\n"

    def stream_response_chars(rag, message):
        """Generate streaming response character by character"""
        try:
            full_response = rag.query_with_context(message)
            
            for char in full_response:
                data = json.dumps({"response": char})
                yield f"data: {data}\n\n"
                time.sleep(0.02)  # 20ms delay between characters
                
        except Exception as e:
            error_data = json.dumps({"error": str(e)})
            yield f"data: {error_data}\n\n"
    
    @app.route('/api/knowledge-bases/<kb_name>', methods=['DELETE'])
    def delete_knowledge_base(kb_name):
        """Delete a knowledge base"""
        try:
            indexer = FileToChromaIndex(persist_dir=CHROMA_FOLDER)
            indexer.delete_knowledge_base(kb_name)
            
            # Also remove from RAG instance cache
            keys_to_remove = [key for key in rag_instances.keys() if key.endswith(f"_{kb_name}")]
            for key in keys_to_remove:
                del rag_instances[key]
            
            # Remove uploaded files directory if it exists
            kb_upload_dir = os.path.join(UPLOAD_FOLDER, kb_name)
            if os.path.exists(kb_upload_dir):
                import shutil
                shutil.rmtree(kb_upload_dir)
            
            return jsonify({"message": f"Knowledge base '{kb_name}' deleted successfully"})
            
        except Exception as e:
            logger.error(f"Error deleting knowledge base: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/knowledge-bases/<kb_name>/info', methods=['GET'])
    def get_knowledge_base_info(kb_name):
        """Get information about a specific knowledge base"""
        try:
            indexer = FileToChromaIndex(persist_dir=CHROMA_FOLDER)
            
            # Load the knowledge base to get info
            indexer.load_knowledge_base(kb_name)
            info = indexer.get_current_collection_info()
            
            return jsonify({
                "name": kb_name,
                "document_count": info.get('count', 0) if info else 0,
                "status": "available"
            })
            
        except Exception as e:
            logger.error(f"Error getting KB info: {e}")
            return jsonify({"error": str(e)}), 404
    
    @app.route('/api/status', methods=['GET'])
    def status():
        """Check if the system is ready"""
        try:
            # Test RAG creation
            test_rag = RAGOllama(model="llama3", chroma_dir=CHROMA_FOLDER)
            
            # Check ChromaDB
            indexer = FileToChromaIndex(persist_dir=CHROMA_FOLDER)
            available_kbs = indexer.list_knowledge_bases()
            
            return jsonify({
                "status": "ready", 
                "message": "RAG system is ready",
                "chroma_dir": CHROMA_FOLDER,
                "available_knowledge_bases": len(available_kbs),
                "knowledge_bases": available_kbs
            })
        except Exception as e:
            logger.error(f"System status check failed: {e}")
            return jsonify({
                "status": "error", 
                "message": f"System not ready: {str(e)}"
            }), 503
    
    return app

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description='RAG Ollama - CLI and Web Interface')
    parser.add_argument('--mode', choices=['cli', 'web'], default='cli',
                       help='Run in CLI mode or start web server (default: cli)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port for web server (default: 5000)')
    parser.add_argument('--host', default='0.0.0.0',
                       help='Host for web server (default: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode for web server')
    
    args = parser.parse_args()
    
    if args.mode == 'web':
        if not FLASK_AVAILABLE:
            print("Error: Flask dependencies not installed.")
            print("Install with: pipenv install")
            print("Or: pip install flask flask-cors")
            return
        
        print("Starting RAG Ollama Web Interface...")
        print("Make sure Ollama is running: ollama serve")
        print(f"Web interface will be available at: http://{args.host}:{args.port}")
        print(f"ChromaDB storage: {os.path.abspath('chroma_store')}")
        
        try:
            app = create_web_app()
            app.run(debug=args.debug, port=args.port, host=args.host)
        except Exception as e:
            print(f"Error starting web server: {e}")
            print("Try running: pipenv install")
    else:
        print("Starting RAG Ollama CLI...")
        chat()

if __name__ == "__main__":
    main()