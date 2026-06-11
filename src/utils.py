"""
Utility Functions for Indian Pharma RAG System

Provides common functions for:
- Input validation
- Text processing
- Error handling
- Formatting
- Logging helpers
"""

import logging
import time
import json
from typing import Any, Dict, List, Optional
from functools import wraps
import re

logger = logging.getLogger(__name__)


# ============================================================================
# VALIDATION HELPERS
# ============================================================================


def validate_query(query: str, min_length: int = 3, max_length: int = 500) -> bool:
    """
    Validate user query.
    
    Args:
        query: User's query string
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    if not query:
        raise ValueError("Query cannot be empty")

    query = query.strip()

    if len(query) < min_length:
        raise ValueError(f"Query must be at least {min_length} characters")

    if len(query) > max_length:
        raise ValueError(f"Query cannot exceed {max_length} characters")

    return True


def validate_api_keys(openai_key: Optional[str], pinecone_key: Optional[str]) -> bool:
    """
    Validate API keys are present.
    
    Args:
        openai_key: OpenAI API key
        pinecone_key: Pinecone API key
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    if not openai_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    if not pinecone_key:
        raise ValueError("PINECONE_API_KEY environment variable is required")

    return True


# ============================================================================
# TEXT PROCESSING
# ============================================================================


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = " ".join(text.split())

    # Remove special characters but keep common punctuation
    text = re.sub(r"[^\w\s\-.,;:!?()₹]", "", text)

    return text.strip()


def extract_drug_name(text: str) -> Optional[str]:
    """
    Extract drug name from text.
    
    Args:
        text: Text containing drug information
        
    Returns:
        Extracted drug name or None
    """
    # Look for patterns like "drug_name: <name>"
    match = re.search(r"(?:drug_name|name|medicine):\s*([^,\n]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return None


def extract_price(text: str) -> Optional[str]:
    """
    Extract price from text.
    
    Args:
        text: Text containing price information
        
    Returns:
        Extracted price or None
    """
    # Look for price patterns (₹ symbol or "Rs", "Rs.", "Price:")
    match = re.search(r"(?:price|cost):\s*([\d,.-]+|[₹Rs.]+[\d,.-]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return None


def format_metadata_display(metadata: Dict[str, Any]) -> str:
    """
    Format metadata for display.
    
    Args:
        metadata: Metadata dictionary
        
    Returns:
        Formatted metadata string
    """
    lines = []
    for key, value in metadata.items():
        if key not in ["text", "ingested_at", "drug_id", "chunk_index"]:
            lines.append(f"  • {key.replace('_', ' ').title()}: {value}")

    return "\n".join(lines)


# ============================================================================
# ERROR HANDLING & RETRIES
# ============================================================================


def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 1.0):
    """
    Decorator for retrying functions with exponential backoff.
    
    RETRY STRATEGY:
    - Handles transient API failures (network, rate limiting)
    - Exponential backoff prevents overwhelming services
    - Logs each retry attempt
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for backoff delay
        
    Example:
        @retry_with_backoff(max_retries=3)
        def call_api():
            ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Max retries exceeded for {func.__name__}")
                        raise

                    wait_time = backoff_factor * (2**attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}. "
                        f"Retrying in {wait_time}s... Error: {e}"
                    )
                    time.sleep(wait_time)

        return wrapper

    return decorator


# ============================================================================
# LOGGING HELPERS
# ============================================================================


def log_api_call(api_name: str, method: str = "GET", **kwargs):
    """
    Log API call details.
    
    Args:
        api_name: Name of the API (e.g., "OpenAI", "Pinecone")
        method: HTTP method used
        **kwargs: Additional details to log
    """
    logger.info(f"{api_name} API call - {method}: {json.dumps(kwargs, default=str)}")


def log_retrieval_results(query: str, num_results: int, avg_score: float):
    """
    Log retrieval statistics.
    
    Args:
        query: User query
        num_results: Number of results returned
        avg_score: Average similarity score
    """
    logger.info(
        f"Retrieval - Query: '{query[:50]}...', Results: {num_results}, Avg Score: {avg_score:.3f}"
    )


# ============================================================================
# FORMATTING HELPERS
# ============================================================================


def format_response(
    answer: str,
    sources: List[Dict[str, Any]],
    query: str,
) -> Dict[str, Any]:
    """
    Format RAG response for API output.
    
    Args:
        answer: Generated answer from LLM
        sources: List of source documents
        query: Original query
        
    Returns:
        Formatted response dictionary
    """
    formatted_sources = []
    for i, source in enumerate(sources, 1):
        formatted_sources.append(
            {
                "id": source.get("id"),
                "relevance_score": round(source.get("score", 0), 3),
                "drug_name": source.get("metadata", {}).get("name"),
                "manufacturer": source.get("metadata", {}).get("manufacturer"),
                "category": source.get("metadata", {}).get("category"),
            }
        )

    return {
        "query": query,
        "answer": answer,
        "sources": formatted_sources,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def format_error_response(error: str, error_type: str = "error") -> Dict[str, Any]:
    """
    Format error response.
    
    Args:
        error: Error message
        error_type: Type of error
        
    Returns:
        Formatted error dictionary
    """
    return {
        "success": False,
        "error": error,
        "error_type": error_type,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str):
        self.name = name
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        duration = self.end - self.start
        logger.info(f"{self.name} took {duration:.3f} seconds")

    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        if self.start and self.end:
            return self.end - self.start
        return 0


# ============================================================================
# BATCH PROCESSING
# ============================================================================


def batch_items(items: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split items into batches.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i : i + batch_size])

    return batches
