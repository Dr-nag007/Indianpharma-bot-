"""
Semantic Search Retriever for Indian Pharma RAG System

This module handles:
1. Converting user queries into embeddings
2. Searching Pinecone vector database semantically
3. Metadata filtering for refined results
4. Result formatting with relevance scores

VECTOR SEARCH EXPLANATION:
- Converts user query into same embedding space as drug documents
- Finds documents with highest cosine similarity to query
- Uses vector distance as relevance score
- Metadata filtering enables targeted searches (e.g., by manufacturer)
"""

import logging
import time
from typing import List, Dict, Any, Optional
import openai
from pinecone import Pinecone

from src.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    TOP_K,
    SIMILARITY_THRESHOLD,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
)

logger = logging.getLogger(__name__)

# Configure OpenAI
openai.api_key = OPENAI_API_KEY


class SemanticRetriever:
    """Retrieves relevant drug information using semantic vector search."""

    def __init__(
        self,
        index_name: str = PINECONE_INDEX_NAME,
        top_k: int = TOP_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ):
        """
        Initialize semantic retriever.
        
        Args:
            index_name: Name of Pinecone index
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score (0-1)
        """
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = index_name
        self.index = self.pc.Index(index_name)
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.embedding_model = EMBEDDING_MODEL

    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Convert query text into embedding vector.
        
        PROCESS:
        1. Query sent to OpenAI embedding API
        2. Returns vector in same space as document embeddings
        3. Enables semantic similarity comparison
        """
        try:
            logger.debug(f"Generating embedding for query: {query}")
            response = openai.Embedding.create(
                input=query,
                model=self.embedding_model,
            )
            embedding = response["data"][0]["embedding"]
            logger.debug(f"Generated embedding with dimension {len(embedding)}")
            return embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise

    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant drug documents.
        
        RETRIEVAL STRATEGY:
        1. Embed the query
        2. Search Pinecone for similar vectors
        3. Apply optional metadata filters
        4. Return top-k results with scores
        
        Args:
            query: User's natural language question
            filters: Optional metadata filters (e.g., {"manufacturer": "Cipla"})
            top_k: Override default top_k value
            
        Returns:
            List of retrieved documents with scores and metadata
        """
        logger.info(f"Searching for query: {query}")

        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []

        try:
            # Get query embedding
            query_embedding = self._get_query_embedding(query)

            # Search Pinecone
            k = top_k or self.top_k
            logger.debug(f"Searching with top_k={k}, filters={filters}")

            search_kwargs = {
                "vector": query_embedding,
                "top_k": k,
                "namespace": PINECONE_NAMESPACE,
                "include_metadata": True,
                "include_values": False,
            }

            if filters:
                search_kwargs["filter"] = filters

            results = self.index.query(**search_kwargs)

            # Format and filter results
            retrieved_docs = []
            for match in results.get("matches", []):
                # Check similarity threshold
                if match["score"] < self.similarity_threshold:
                    logger.debug(
                        f"Skipping result with score {match['score']} below threshold {self.similarity_threshold}"
                    )
                    continue

                doc = {
                    "id": match["id"],
                    "score": match["score"],  # Similarity score 0-1
                    "metadata": match.get("metadata", {}),
                    "text": match.get("metadata", {}).get("text", ""),
                }
                retrieved_docs.append(doc)

            logger.info(f"Retrieved {len(retrieved_docs)} relevant documents")
            return retrieved_docs

        except Exception as e:
            logger.error(f"Search error: {e}")
            raise

    def search_by_drug_name(self, drug_name: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for specific drug by name with exact matching.
        
        Args:
            drug_name: Name of the drug
            top_k: Number of results to return
            
        Returns:
            Documents matching the drug name
        """
        logger.info(f"Searching for drug: {drug_name}")

        # Create filter for exact drug name match
        filters = {"name": {"$eq": drug_name}}

        return self.search(f"Information about {drug_name}", filters=filters, top_k=top_k)

    def search_by_manufacturer(
        self, query: str, manufacturer: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for drugs from specific manufacturer.
        
        METADATA FILTERING:
        - Filters results by manufacturer before returning
        - Reduces irrelevant results
        - Enables refined queries
        
        Args:
            query: Search query
            manufacturer: Manufacturer name to filter by
            top_k: Number of results to return
            
        Returns:
            Documents from specified manufacturer
        """
        logger.info(f"Searching for '{query}' from manufacturer: {manufacturer}")

        filters = {"manufacturer": {"$eq": manufacturer}}

        return self.search(query, filters=filters, top_k=top_k)

    def search_by_category(
        self, query: str, category: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for drugs in specific category.
        
        Args:
            query: Search query
            category: Drug category to filter by
            top_k: Number of results to return
            
        Returns:
            Documents from specified category
        """
        logger.info(f"Searching for '{query}' in category: {category}")

        filters = {"category": {"$eq": category}}

        return self.search(query, filters=filters, top_k=top_k)

    def format_retrieval_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context string for LLM.
        
        Args:
            retrieved_docs: List of documents from search
            
        Returns:
            Formatted context string with source attribution
        """
        if not retrieved_docs:
            return "No relevant information found in database."

        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            # Add relevance indicator
            relevance = int(doc["score"] * 100)

            # Build context section
            context_parts.append(f"Source {i} (Relevance: {relevance}%):")
            context_parts.append(doc["text"])

            # Add metadata if available
            metadata = doc.get("metadata", {})
            if metadata:
                context_parts.append(f"Details: {metadata.get('category', 'Unknown')} - {metadata.get('manufacturer', 'Unknown')}")

            context_parts.append("-" * 50)

        return "\n".join(context_parts)


class HybridRetriever:
    """Combines semantic search with metadata filtering for better results."""

    def __init__(self):
        """Initialize hybrid retriever with semantic component."""
        self.semantic_retriever = SemanticRetriever()

    def search_with_refinement(
        self,
        query: str,
        refine_by: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Two-stage retrieval: semantic search + optional refinement.
        
        HYBRID RETRIEVAL:
        1. First stage: Semantic search on query
        2. Second stage: Refine results by metadata (optional)
        3. Combines best of both approaches
        
        Args:
            query: Main search query
            refine_by: Optional filters like {"manufacturer": "Cipla", "category": "Antibiotic"}
            
        Returns:
            Refined list of documents
        """
        logger.info(f"Hybrid search - Query: {query}, Refine by: {refine_by}")

        # Stage 1: Semantic search
        results = self.semantic_retriever.search(query)

        if not refine_by:
            return results

        # Stage 2: Metadata filtering
        refined_results = []
        for doc in results:
            metadata = doc.get("metadata", {})
            matches_filter = True

            for key, value in refine_by.items():
                if metadata.get(key) != value:
                    matches_filter = False
                    break

            if matches_filter:
                refined_results.append(doc)

        logger.info(f"Refined to {len(refined_results)} results after filtering")
        return refined_results

    def get_formatted_context(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Get formatted context for RAG pipeline.
        
        Returns:
            Tuple of (formatted_context, retrieved_documents)
        """
        docs = self.search_with_refinement(query)
        context = self.semantic_retriever.format_retrieval_context(docs)
        return context, docs


# Type hint for convenience
from typing import Tuple
