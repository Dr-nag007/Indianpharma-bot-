"""
FastAPI Backend for Indian Pharma RAG System

REST API endpoints for:
- Query processing (POST /query)
- Data ingestion (POST /ingest)
- Health checks (GET /health)
- System statistics (GET /stats)

DEPLOYMENT READY:
- Async request handling
- Comprehensive error handling
- Request validation
- CORS support
- Logging for all operations
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import logging
import time
from datetime import datetime

from src.rag_pipeline import IndianPharmaRAGPipeline
from src.ingest import ingest_dataset
from src.utils import validate_query, format_error_response, Timer

logger = logging.getLogger(__name__)

# ============================================================================
# FASTAPI SETUP
# ============================================================================

app = FastAPI(
    title="Indian Pharma RAG API",
    description="Retrieval-Augmented Generation system for Indian pharmaceutical products",
    version="1.0.0",
)

# Add CORS middleware for web app support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline
rag_pipeline = IndianPharmaRAGPipeline()

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================


class QueryRequest(BaseModel):
    """Request model for drug queries."""

    query: str = Field(..., min_length=3, max_length=500, description="User's question about drugs")
    include_sources: bool = Field(
        True,
        description="Whether to include source documents in response",
    )
    top_k: Optional[int] = Field(
        5,
        ge=1,
        le=10,
        description="Number of documents to retrieve",
    )

    @validator("query")
    def validate_query_field(cls, v):
        """Validate query field."""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()


class IngestRequest(BaseModel):
    """Request model for data ingestion."""

    filepath: str = Field(..., description="Path to CSV/Excel/JSON file to ingest")
    dataset_name: Optional[str] = Field(None, description="Name of the dataset")

    @validator("filepath")
    def validate_filepath(cls, v):
        """Validate filepath is not empty."""
        if not v.strip():
            raise ValueError("Filepath cannot be empty")
        return v.strip()


class Source(BaseModel):
    """Source document reference."""

    id: str
    relevance_score: float
    drug_name: Optional[str]
    manufacturer: Optional[str]
    category: Optional[str]


class QueryResponse(BaseModel):
    """Response model for queries."""

    query: str
    answer: str
    sources: List[Source]
    timestamp: str
    success: bool = True


class HealthResponse(BaseModel):
    """Response model for health checks."""

    status: str
    timestamp: str
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Response model for errors."""

    success: bool = False
    error: str
    error_type: str
    timestamp: str


class IngestResponse(BaseModel):
    """Response model for ingestion."""

    success: bool
    message: str
    dataset_name: str
    timestamp: str


class StatsResponse(BaseModel):
    """Response model for statistics."""

    status: str
    uptime_seconds: float
    total_queries_processed: int
    timestamp: str


# ============================================================================
# GLOBAL STATE (for demo purposes - use Redis in production)
# ============================================================================

app_state = {
    "start_time": time.time(),
    "queries_processed": 0,
    "errors": 0,
}


# ============================================================================
# ENDPOINTS
# ============================================================================


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    
    Returns system status for deployment monitoring.
    Used by load balancers, orchestration tools (Kubernetes, Docker Compose).
    """
    logger.info("Health check requested")
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
    }


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    """
    Get system statistics.
    
    Returns:
        Uptime, queries processed, error count
    """
    uptime = time.time() - app_state["start_time"]
    return {
        "status": "running",
        "uptime_seconds": uptime,
        "total_queries_processed": app_state["queries_processed"],
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/query", response_model=QueryResponse, tags=["RAG Pipeline"])
async def process_query(request: QueryRequest) -> Dict[str, Any]:
    """
    Process a query about Indian drugs.
    
    PIPELINE:
    1. Validate query
    2. Retrieve relevant drug documents
    3. Generate answer using GPT
    4. Return formatted response with sources
    
    Args:
        request: QueryRequest with question and options
        
    Returns:
        QueryResponse with answer and source references
        
    Examples:
        - "What is Aspirin used for?"
        - "What are the side effects of Crocin?"
        - "Which manufacturer makes Amoxicillin in India?"
    """
    logger.info(f"Query received: {request.query[:100]}...")
    app_state["queries_processed"] += 1

    try:
        with Timer("Complete Query Processing"):
            # Process query through RAG pipeline
            result = rag_pipeline.query(request.query, top_k=request.top_k or 5)

            # Filter sources if needed
            if not request.include_sources:
                result["sources"] = []

            # Ensure success flag
            result["success"] = True

            logger.info("Query processed successfully")
            return result

    except ValueError as e:
        app_state["errors"] += 1
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        app_state["errors"] += 1
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during query processing")


@app.post("/ingest", response_model=IngestResponse, tags=["Data Management"], background_tasks=BackgroundTasks)
async def ingest_data(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Ingest new pharmaceutical data.
    
    INGESTION PROCESS:
    1. Load CSV/Excel/JSON file
    2. Clean and validate data
    3. Generate embeddings
    4. Upsert to Pinecone
    5. Return status
    
    Args:
        request: IngestRequest with file path
        background_tasks: FastAPI background task manager
        
    Returns:
        IngestResponse with success status
        
    Note:
        Long-running ingestions are processed in background tasks.
    """
    logger.info(f"Ingestion request for: {request.filepath}")

    try:
        # Start ingestion in background
        background_tasks.add_task(ingest_dataset, request.filepath)

        return {
            "success": True,
            "message": f"Ingestion started for {request.filepath}. Processing in background.",
            "dataset_name": request.dataset_name or request.filepath,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=400, detail=f"Ingestion failed: {str(e)}")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Indian Pharma RAG API",
        "version": "1.0.0",
        "description": "Retrieval-Augmented Generation for Indian pharmaceutical products",
        "endpoints": {
            "health": "GET /health",
            "stats": "GET /stats",
            "query": "POST /query",
            "ingest": "POST /ingest",
            "docs": "GET /docs",
        },
    }


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    logger.error(f"HTTP exception: {exc.detail}")
    return {
        "success": False,
        "error": exc.detail,
        "error_type": "http_error",
        "timestamp": datetime.now().isoformat(),
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return {
        "success": False,
        "error": "Internal server error",
        "error_type": "server_error",
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================


@app.on_event("startup")
async def startup_event():
    """Initialize on application startup."""
    logger.info("Indian Pharma RAG API starting up...")
    logger.info(f"RAG Pipeline initialized: {rag_pipeline.llm_model}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Indian Pharma RAG API shutting down...")
    uptime = time.time() - app_state["start_time"]
    logger.info(f"Uptime: {uptime:.2f} seconds")
    logger.info(f"Queries processed: {app_state['queries_processed']}")
    logger.info(f"Errors: {app_state['errors']}")


# ============================================================================
# RUN SERVER (for development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=4,
        log_level="info",
    )
