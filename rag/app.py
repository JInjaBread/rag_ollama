import logging
from rag.module.RagOllama import RAGOllama

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

def chat():
    logger.info("Starting RAG Chat with NovaTech Knowledge Base")
    print("RAG Chat with NovaTech Knowledge Base (type 'exit' to quit)")

    rag = RAGOllama(model="llama3")
    logger.info("Building index from novatech_company_profile.pdf")
    rag.build_index("novatech_company_profile.pdf")

    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            logger.info("User exited the chat.")
            print("Goodbye!")
            break
        
        logger.info(f"User query: {query}")
        answer = rag.query_with_context(query)
        logger.info(f"Ollama answer: {answer}")