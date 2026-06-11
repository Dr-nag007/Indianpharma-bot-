"""
Data Ingestion Module for Indian Pharma RAG System

This module handles:
1. Loading Indian pharmaceutical datasets from CSV/Excel/JSON
2. Data cleaning and validation
3. Text chunking strategies
4. Batch embedding generation using OpenAI
5. Upserting embeddings to Pinecone vector database

ARCHITECTURE:
- Data Loading: Reads multiple formats (CSV, Excel, JSON)
- Cleaning: Handles missing values, standardizes field names
- Chunking: Splits text by overlapping windows for better retrieval
- Embedding: Batch processes text through OpenAI's embedding API
- Upserting: Efficiently stores vectors in Pinecone with metadata
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import time
from datetime import datetime
import openai
from pinecone import Pinecone

from src.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    EMBEDDING_DIMENSION,
    BATCH_SIZE,
    DATA_DIR,
)

logger = logging.getLogger(__name__)

# Configure OpenAI
openai.api_key = OPENAI_API_KEY


class IndianDrugDataProcessor:
    """Loads and processes Indian pharmaceutical data with cleaning and validation."""

    # Standard columns we expect in drug datasets
    STANDARD_COLUMNS = {
        "name": ["drug_name", "product_name", "medicine_name", "name"],
        "generic_name": ["generic_name", "salt_composition", "active_ingredient"],
        "manufacturer": ["manufacturer", "company", "manufacturer_name"],
        "category": ["category", "type", "drug_type"],
        "dosage": ["dosage", "dose", "strength"],
        "indication": ["indication", "uses", "usage"],
        "side_effects": ["side_effects", "adverse_effects", "contraindications"],
        "price": ["price", "cost", "mrp"],
        "composition": ["composition", "salt_composition", "active_ingredients"],
    }

    def __init__(self):
        """Initialize the data processor."""
        self.data: Optional[pd.DataFrame] = None
        self.processed_docs: List[Dict[str, Any]] = []

    def load_csv(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            logger.info(f"Loading CSV data from {filepath}")
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} rows from CSV")
            self.data = df
            return df
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise

    def load_excel(self, filepath: str, sheet_name: int = 0) -> pd.DataFrame:
        """Load data from Excel file."""
        try:
            logger.info(f"Loading Excel data from {filepath}")
            df = pd.read_excel(filepath, sheet_name=sheet_name)
            logger.info(f"Loaded {len(df)} rows from Excel")
            self.data = df
            return df
        except Exception as e:
            logger.error(f"Error loading Excel: {e}")
            raise

    def load_json(self, filepath: str) -> pd.DataFrame:
        """Load data from JSON file."""
        try:
            logger.info(f"Loading JSON data from {filepath}")
            df = pd.read_json(filepath)
            logger.info(f"Loaded {len(df)} rows from JSON")
            self.data = df
            return df
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            raise

    def clean_data(self) -> pd.DataFrame:
        """Clean and standardize loaded data."""
        if self.data is None:
            raise ValueError("No data loaded. Use load_csv/load_excel/load_json first.")

        df = self.data.copy()
        logger.info("Starting data cleaning process...")

        # Step 1: Standardize column names to lowercase
        df.columns = df.columns.str.lower().str.strip()
        logger.info(f"Standardized column names: {list(df.columns)}")

        # Step 2: Normalize column names by mapping variants
        df = self._normalize_columns(df)

        # Step 3: Remove rows where core fields are empty
        core_fields = ["name"]
        df = df.dropna(subset=core_fields)
        logger.info(f"Removed rows with missing core fields. Remaining: {len(df)}")

        # Step 4: Fill other missing values with sensible defaults
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna("Not specified")
            else:
                df[col] = df[col].fillna(0)

        # Step 5: Remove duplicates based on name and manufacturer
        initial_len = len(df)
        df = df.drop_duplicates(subset=["name", "manufacturer"], keep="first")
        logger.info(f"Removed {initial_len - len(df)} duplicate entries")

        self.data = df
        logger.info(f"Data cleaning complete. Final dataset: {len(df)} drugs")
        return df

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map variant column names to standard names."""
        column_mapping = {}
        for standard_name, variants in self.STANDARD_COLUMNS.items():
            for variant in variants:
                if variant in df.columns:
                    column_mapping[variant] = standard_name
                    break

        df = df.rename(columns=column_mapping)
        return df

    def chunk_drug_record(self, record: Dict[str, Any]) -> List[str]:
        """
        Split a drug record into chunks with overlap for better retrieval.
        
        CHUNKING STRATEGY:
        - Combines multiple fields into a formatted document
        - Splits into fixed-size overlapping chunks
        - Preserves semantic meaning within chunks
        - Overlap ensures no important info is lost at boundaries
        """
        # Format the drug record as text
        text_parts = []

        # Prioritize important fields
        priority_fields = ["name", "generic_name", "manufacturer", "dosage", "indication"]
        other_fields = [k for k in record.keys() if k not in priority_fields]

        # Add priority fields first
        for field in priority_fields:
            if field in record and pd.notna(record[field]):
                text_parts.append(f"{field}: {record[field]}")

        # Add other fields
        for field in other_fields:
            if pd.notna(record[field]):
                text_parts.append(f"{field}: {record[field]}")

        full_text = "\n".join(text_parts)

        # Chunk the text with overlap
        chunks = []
        text_length = len(full_text)

        if text_length <= CHUNK_SIZE:
            chunks.append(full_text)
        else:
            # Create overlapping chunks
            for i in range(0, text_length, CHUNK_SIZE - CHUNK_OVERLAP):
                chunk = full_text[i : i + CHUNK_SIZE]
                if chunk.strip():
                    chunks.append(chunk)

                # Stop if we've reached the end
                if i + CHUNK_SIZE >= text_length:
                    break

        return chunks

    def process_dataset(self) -> List[Dict[str, Any]]:
        """
        Process entire dataset into documents ready for embedding.
        
        Returns:
            List of documents with content and metadata
        """
        if self.data is None:
            raise ValueError("No data to process. Load and clean data first.")

        logger.info("Processing dataset into documents...")
        self.processed_docs = []
        doc_id = 0

        for idx, row in self.data.iterrows():
            record = row.to_dict()
            chunks = self.chunk_drug_record(record)

            for chunk_idx, chunk_content in enumerate(chunks):
                doc = {
                    "id": f"drug_{idx}_chunk_{chunk_idx}",
                    "content": chunk_content,
                    "metadata": {
                        "drug_id": idx,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks),
                        "source": "pharma_dataset",
                        "ingested_at": datetime.now().isoformat(),
                        **{k: str(v) for k, v in record.items()},
                    },
                }
                self.processed_docs.append(doc)
                doc_id += 1

        logger.info(f"Generated {len(self.processed_docs)} documents from dataset")
        return self.processed_docs


class EmbeddingGenerator:
    """Generates embeddings for drug documents using OpenAI API."""

    def __init__(self, model: str = EMBEDDING_MODEL, batch_size: int = BATCH_SIZE):
        """
        Initialize embedding generator.
        
        EMBEDDING EXPLANATION:
        - Converts text into numerical vectors (dense embeddings)
        - Similar drugs/concepts get similar vectors
        - Enables semantic similarity search without keyword matching
        - text-embedding-3-small: 1536 dimensions, fast and cost-effective
        """
        self.model = model
        self.batch_size = batch_size
        self.embeddings_cache: Dict[str, List[float]] = {}

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI.
        
        BATCH PROCESSING:
        - Processes multiple texts in single API call
        - More efficient than individual requests
        - Reduces costs and latency
        """
        logger.info(f"Generating embeddings for {len(texts)} texts")

        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            logger.info(f"Processing batch {i // self.batch_size + 1}...")

            try:
                response = openai.Embedding.create(
                    input=batch,
                    model=self.model,
                )

                for j, item in enumerate(response["data"]):
                    embeddings.append(item["embedding"])

                # Add small delay to avoid rate limits
                if i + self.batch_size < len(texts):
                    time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                raise

        logger.info(f"Successfully generated {len(embeddings)} embeddings")
        return embeddings

    def generate_embedding_for_text(self, text: str) -> List[float]:
        """Generate a single embedding for a text."""
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]

        try:
            response = openai.Embedding.create(
                input=text,
                model=self.model,
            )
            embedding = response["data"][0]["embedding"]
            self.embeddings_cache[text] = embedding
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise


class PineconeUpserter:
    """Handles upserting documents and embeddings to Pinecone."""

    def __init__(self, index_name: str = PINECONE_INDEX_NAME):
        """Initialize Pinecone upserter."""
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = index_name
        self.index = self.pc.Index(index_name)
        self._ensure_index_exists()

    def _ensure_index_exists(self):
        """Create index if it doesn't exist."""
        try:
            indexes = self.pc.list_indexes()
            existing_names = [idx.name for idx in indexes]

            if self.index_name not in existing_names:
                logger.info(f"Creating Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=EMBEDDING_DIMENSION,
                    metric="cosine",
                    spec={
                        "serverless": {
                            "cloud": "aws",
                            "region": "us-east-1",
                        }
                    },
                )
                logger.info("Index created successfully")
            else:
                logger.info(f"Index {self.index_name} already exists")
        except Exception as e:
            logger.error(f"Error ensuring index exists: {e}")
            raise

    def upsert_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Upsert documents with embeddings to Pinecone.
        
        UPSERTING EXPLANATION:
        - "Upsert" = Update if exists, Insert if new
        - Stores vectors with metadata for later retrieval
        - Metadata enables filtering and source tracking
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Document count ({len(documents)}) must match embedding count ({len(embeddings)})"
            )

        logger.info(f"Upserting {len(documents)} documents to Pinecone...")

        # Prepare vectors for upsert
        vectors = []
        for doc, embedding in zip(documents, embeddings):
            vectors.append(
                (
                    doc["id"],
                    embedding,
                    {
                        "text": doc["content"][:1000],  # Store first 1000 chars
                        **doc["metadata"],
                    },
                )
            )

        # Upsert in batches to Pinecone
        try:
            for i in range(0, len(vectors), self.batch_size):
                batch = vectors[i : i + self.batch_size]
                self.index.upsert(
                    vectors=batch,
                    namespace=PINECONE_NAMESPACE,
                )
                logger.info(f"Upserted batch {i // self.batch_size + 1}")

            logger.info("All documents upserted successfully")
        except Exception as e:
            logger.error(f"Error upserting documents: {e}")
            raise

    @property
    def batch_size(self) -> int:
        """Get batch size for upserts."""
        return BATCH_SIZE


def ingest_dataset(filepath: str):
    """
    Complete ingestion pipeline: Load -> Clean -> Embed -> Upsert to Pinecone.
    
    This is the main function to run for ingesting new pharma datasets.
    """
    logger.info("=" * 80)
    logger.info("STARTING PHARMA DATA INGESTION PIPELINE")
    logger.info("=" * 80)

    try:
        # Step 1: Load and clean data
        processor = IndianDrugDataProcessor()

        if filepath.endswith(".csv"):
            processor.load_csv(filepath)
        elif filepath.endswith((".xlsx", ".xls")):
            processor.load_excel(filepath)
        elif filepath.endswith(".json"):
            processor.load_json(filepath)
        else:
            raise ValueError("Unsupported file format. Use CSV, Excel, or JSON.")

        processor.clean_data()

        # Step 2: Process into documents
        documents = processor.process_dataset()

        # Step 3: Generate embeddings
        embedding_gen = EmbeddingGenerator()
        texts = [doc["content"] for doc in documents]
        embeddings = embedding_gen.generate_embeddings(texts)

        # Step 4: Upsert to Pinecone
        upserter = PineconeUpserter()
        upserter.upsert_documents(documents, embeddings)

        logger.info("=" * 80)
        logger.info("INGESTION PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Ingestion pipeline failed: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        ingest_dataset(filepath)
    else:
        print("Usage: python ingest.py <filepath>")
        print("Example: python ingest.py data/drugs.csv")
