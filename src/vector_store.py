from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages Pinecone vector store for drug database."""

    def __init__(
        self,
        api_key: str,
        environment: str,
        index_name: str,
        embedding_model: str,
        namespace: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.namespace = namespace or ""
        self.embedding_model = embedding_model

        self.pc = Pinecone(api_key=api_key)
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=openai_api_key,
        )

        self._ensure_index_exists()
        self.index = self.pc.Index(index_name)

    def _ensure_index_exists(self):
        try:
            existing_indexes = self.pc.indexes.list().names()
            if self.index_name not in existing_indexes:
                logger.info(f"Creating index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                logger.info(f"Index {self.index_name} created successfully")
            else:
                logger.info(f"Index {self.index_name} already exists")
        except Exception as e:
            logger.error(f"Error managing index: {e}")
            raise

    def add_documents(self, documents: List[Document]) -> None:
        try:
            # Chunk long documents to improve retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50
            )

            # Prepare chunks and metadata
            chunk_texts = []
            chunk_metadatas = []
            for doc_idx, doc in enumerate(documents):
                chunks = text_splitter.split_text(doc.page_content or "")
                for chunk_idx, chunk in enumerate(chunks):
                    metadata = {**(doc.metadata or {}), "page_content": chunk, "original_id": doc_idx, "chunk_index": chunk_idx}
                    chunk_texts.append(chunk)
                    chunk_metadatas.append(metadata)

            # Generate embeddings for all chunks
            embeddings = self.embeddings.embed_documents(chunk_texts)

            vectors = []
            for idx, (vector, metadata) in enumerate(zip(embeddings, chunk_metadatas)):
                vector_id = f"{self.index_name}-{idx}"
                vectors.append((vector_id, vector, metadata))

            # Upsert vectors into Pinecone
            self.index.upsert(
                vectors=vectors,
                namespace=self.namespace,
                show_progress=False,
            )
            logger.info(f"Added {len(chunk_texts)} document chunks to vector store")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    def query(self, query_text: str, top_k: int = 5) -> List[Document]:
        try:
            query_embedding = self.embeddings.embed_query(query_text)
            response = self.index.query(
                top_k=top_k,
                vector=query_embedding,
                include_metadata=True,
                namespace=self.namespace,
            )

            documents = []
            for match in getattr(response, "matches", []) or []:
                metadata = match.metadata or {}
                content = metadata.get("page_content", "")
                documents.append(Document(page_content=content, metadata=metadata))
            return documents
        except Exception as e:
            logger.error(f"Error querying vector store: {e}")
            raise

    # Alias for similarity search (convenience)
    def similarity_search(self, query_text: str, top_k: int = 5) -> List[Document]:
        return self.query(query_text, top_k=top_k)

    def get_vector_store(self) -> "VectorStoreManager":
        return self

    def delete_all(self) -> None:
        try:
            self.index.delete(delete_all=True, namespace=self.namespace)
            logger.info("All vectors deleted from index")
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            raise
