import logging
from src.config import (
    OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME, PINECONE_NAMESPACE, EMBEDDING_MODEL,
    LLM_MODEL, TOP_K
)
from src.data_loader import IndianDrugDataLoader
from src.vector_store import VectorStoreManager
from src.rag_pipeline import DrugRAGPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def initialize_rag_system():
    """Initialize and setup the RAG system"""
    
    try:
        logger.info("Initializing Indian Drug RAG System...")
        
        # 1. Load data
        logger.info("Loading drug data...")
        loader = IndianDrugDataLoader(data_dir="data")
        
        # Load from your Indian medicine dataset in the repository data folder
        documents = loader.load_from_csv("data/indian_medicine_data.csv")
        logger.info(f"Successfully loaded {len(documents)} drugs from dataset")
        
        # 2. Setup vector store
        logger.info("Setting up vector store...")
        vector_store_manager = VectorStoreManager(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENVIRONMENT,
            index_name=PINECONE_INDEX_NAME,
            embedding_model=EMBEDDING_MODEL,
            namespace=PINECONE_NAMESPACE,
            openai_api_key=OPENAI_API_KEY,
        )
        
        # Add documents to vector store
        vector_store_manager.add_documents(documents)
        
        # Get configured vector store
        vector_store = vector_store_manager.get_vector_store()
        
        # 3. Create RAG pipeline
        logger.info("Creating RAG pipeline...")
        rag_pipeline = DrugRAGPipeline(
            vector_store=vector_store,
            llm_model=LLM_MODEL,
            openai_api_key=OPENAI_API_KEY,
            top_k=TOP_K
        )
        
        logger.info("RAG System initialized successfully!")
        return rag_pipeline
    
    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")
        raise


def demo_queries(rag_pipeline: DrugRAGPipeline):
    """Run demo queries"""
    
    demo_questions = [
        "What is Augmentin used for and who manufactures it?",
        "Tell me about Azithral and its composition",
        "Which antibiotics are available in India?",
        "What is the price range of allergy medicines?",
        "Tell me about drugs containing Paracetamol",
    ]
    
    logger.info("\n" + "="*60)
    logger.info("Running Demo Queries")
    logger.info("="*60 + "\n")
    
    for question in demo_questions:
        logger.info(f"Q: {question}")
        result = rag_pipeline.query(question)
        logger.info(f"A: {result['answer']}\n")
        
        if result['sources']:
            logger.info(f"Sources: {len(result['sources'])} documents retrieved\n")


def run_chatbot(rag_pipeline: DrugRAGPipeline):
    """Run the RAG system in chatbot mode."""
    logger.info("\n" + "="*60)
    logger.info("Chatbot Mode - Ask questions about Indian drugs")
    logger.info("Type 'exit' to quit")
    logger.info("="*60 + "\n")
    
    chat_history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            logger.info("Exiting chatbot mode...")
            break
        if not user_input:
            continue

        result = rag_pipeline.chat(user_input, chat_history)
        answer = result.get('answer', 'Sorry, I could not generate an answer.')
        print(f"\nAssistant: {answer}\n")

        if result['sources']:
            print("Sources:")
            for idx, source in enumerate(result['sources'], start=1):
                print(f"  {idx}. {source['metadata'].get('name', source['metadata'].get('drug_name', 'unknown'))}")
            print()

        chat_history.append((user_input, answer))


if __name__ == "__main__":
    try:
        # Initialize RAG system
        rag_pipeline = initialize_rag_system()
        
        # Run demo queries
        demo_queries(rag_pipeline)
        
        # Launch chatbot mode
        run_chatbot(rag_pipeline)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
