"""
Production Configuration for Indian Pharma RAG System

This module loads and validates all configuration from environment variables.
It manages:
- OpenAI API settings for embeddings and LLM generation
- Pinecone settings for vector database operations
- RAG pipeline parameters (chunking, retrieval strategy)
- Logging and deployment settings
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# OPENAI CONFIGURATION
# ============================================================================
# OpenAI API Key - Required for embeddings and LLM generation
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Embedding Model - Generates vector representations of text
# text-embedding-3-small: Fast, cheap, 1536 dimensions
# text-embedding-3-large: More accurate, 3072 dimensions
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

# LLM Model - Generates final answers from retrieved context
# gpt-3.5-turbo: Fast and cost-effective
# gpt-4: More capable but slower and expensive
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# ============================================================================
# PINECONE CONFIGURATION
# ============================================================================
# Pinecone Vector Database - Stores and retrieves drug embeddings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is required")

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "indian-pharma-index")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "drug-data")
PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")

# ============================================================================
# RAG PIPELINE CONFIGURATION
# ============================================================================
# Chunking Strategy
# How text is split into manageable pieces for embedding and retrieval
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))  # Characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # Overlap between chunks

# Retrieval Strategy
# How many top matches to retrieve and use as context
TOP_K = int(os.getenv("TOP_K", "5"))  # Number of document chunks to retrieve
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_NAME = os.getenv("PROJECT_NAME", "indian-pharma-rag")
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"

# Create necessary directories
for directory in [DATA_DIR, LOGS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DEPLOYMENT CONFIGURATION
# ============================================================================
# Environment detection for deployment optimization
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")  # development, staging, production
DEBUG = ENVIRONMENT == "development"

# API Server Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "4"))

# Batch Processing Configuration (for efficient bulk operations)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))  # Size of batches for bulk embeddings
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "60"))  # Requests per minute

# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOGS_DIR / "app.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)
logger.info(f"Configuration loaded for {ENVIRONMENT} environment")
