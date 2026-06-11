"""
RAG Pipeline for Indian Pharma Question Answering

This module implements the core RAG (Retrieval-Augmented Generation) pipeline:

RAG WORKFLOW:
1. USER QUERY
   - User asks a question about Indian drugs
   
2. RETRIEVAL
   - Query is embedded using OpenAI
   - Semantic search finds relevant drug documents
   - Top-K documents are retrieved from Pinecone
   
3. CONTEXT BUILDING
   - Retrieved documents are formatted as context
   - Conversation history is included if available
   
4. GENERATION
   - Context + History + Query sent to GPT
   - LLM generates informed answer based on retrieved context
   - Response includes source attributions
   
5. OUTPUT
   - Final answer with source references
   - Confidence scores and metadata
"""

import logging
import openai
from typing import Optional, List, Dict, Any, Tuple
import time

from src.config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)
from src.utils import validate_query, format_response, format_error_response, Timer

logger = logging.getLogger(__name__)

# Configure OpenAI
openai.api_key = OPENAI_API_KEY


class IndianPharmaRAGPipeline:
    """
    Complete RAG pipeline for Indian pharmaceutical Q&A.
    
    Combines semantic retrieval with LLM generation for accurate,
    context-aware answers about Indian drugs.
    """

    # System prompt for LLM - defines behavior and constraints
    SYSTEM_PROMPT = """You are an expert Indian pharmaceutical specialist with deep knowledge of:
- Drugs available in the Indian market
- Generic and brand names
- Manufacturing companies
- Dosages and formulations
- Side effects and contraindications
- Pricing in Indian rupees
- Usage guidelines and indications

INSTRUCTIONS:
1. Answer based ONLY on the provided context from the drug database
2. Cite specific drug information: name, manufacturer, dosage, price
3. Always mention important side effects and contraindications
4. If information is not in the database, clearly state this
5. Recommend consulting healthcare professionals for medical decisions
6. Be accurate and conservative in medical claims
7. Provide alternative drugs when relevant

FORMAT:
- Start with the most relevant drug information
- Organize by: Name → Manufacturer → Dosage → Uses → Side Effects → Price
- Always cite your sources from the provided context
- End with a clear recommendation to consult doctors if needed"""

    def __init__(
        self,
        vector_store=None,
        llm_model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        top_k: int = 5,
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            llm_model: Which GPT model to use
            temperature: Creativity level (0.0=deterministic, 1.0=creative)
            max_tokens: Maximum response length
            top_k: Number of documents to retrieve
        """
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k

        # Attach provided vector store (expects `similarity_search` method)
        self.vector_store = vector_store

        logger.info(f"RAG Pipeline initialized with model: {llm_model}")

    def _retrieve_context(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retrieve relevant context from vector database.
        
        RETRIEVAL PROCESS:
        1. Convert query to embedding
        2. Search Pinecone for similar documents
        3. Format documents as context
        4. Return context + metadata
        
        Args:
            query: User's natural language question
            
        Returns:
            Tuple of (formatted_context, retrieved_documents)
        """
        try:
            with Timer("Semantic Retrieval"):
                logger.info(f"Retrieving context for query: {query[:100]}...")

                # Use provided vector store's similarity_search if available
                if not self.vector_store:
                    raise RuntimeError("No vector_store configured for retrieval")

                retrieved_docs = self.vector_store.similarity_search(query, top_k=self.top_k)

                if not retrieved_docs:
                    logger.warning("No documents retrieved for query")
                    return (
                        "No relevant information found in the Indian drug database.",
                        [],
                    )

                # Build context from Document objects
                context_parts = []
                for i, doc in enumerate(retrieved_docs, 1):
                    metadata = doc.metadata or {}
                    context_parts.append(f"Source {i} (id={metadata.get('original_id','unknown')}):")
                    context_parts.append(doc.page_content)
                    context_parts.append(f"Metadata: {metadata}")
                    context_parts.append("-" * 50)

                context = "\n".join(context_parts)
                logger.info(f"Retrieved {len(retrieved_docs)} documents")

                return context, retrieved_docs

        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            raise

    def _build_prompt(
        self,
        context: str,
        question: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
    ) -> str:
        """
        Build the complete prompt for the LLM.
        
        PROMPT ENGINEERING:
        - Includes system instructions
        - Adds retrieval context
        - Embeds conversation history
        - Formats question clearly
        
        Args:
            context: Retrieved document context
            question: User's question
            chat_history: Previous conversation turns
            
        Returns:
            Complete prompt for LLM
        """
        prompt_parts = []

        # Add context — explicitly instruct model to use ONLY this context
        prompt_parts.append("=== DRUG DATABASE CONTEXT (USE ONLY THIS) ===")
        prompt_parts.append(context)
        prompt_parts.append("")

        # Add chat history if available
        if chat_history:
            prompt_parts.append("=== CONVERSATION HISTORY ===")
            for user_msg, assistant_msg in chat_history[-3:]:  # Last 3 turns
                prompt_parts.append(f"User: {user_msg}")
                prompt_parts.append(f"Assistant: {assistant_msg}")
            prompt_parts.append("")

        # Add current question
        prompt_parts.append("=== CURRENT QUESTION ===")
        prompt_parts.append(question)

        return "\n".join(prompt_parts)

    def _generate_response(self, context: str, question: str) -> str:
        """
        Generate answer using OpenAI GPT.
        
        LLM GENERATION STRATEGY:
        - Uses GPT-3.5-turbo (fast, cost-effective)
        - Temperature 0.3 (mostly deterministic)
        - Max 1024 tokens (reasonable response length)
        - Includes system prompt with pharmaceutical expertise
        
        Args:
            context: Retrieved context from vector DB
            question: User's question
            
        Returns:
            Generated answer
        """
        try:
            logger.debug("Calling OpenAI API for answer generation")

            with Timer("LLM Generation"):
                prompt = self._build_prompt(context, question)

                # Ensure model is only using retrieved context — low temperature
                response = openai.ChatCompletion.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT + "\nDO NOT HALLUCINATE; if answer not in context, say so."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=max(0.0, min(0.3, self.temperature)),
                    max_tokens=self.max_tokens,
                    top_p=0.95,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                )

                answer = response.choices[0].message.content.strip()
                logger.info(f"Generated response ({len(answer)} chars)")

                return answer

        except openai.error.RateLimitError:
            logger.error("Rate limit exceeded. Please try again later.")
            raise
        except openai.error.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def query(
        self,
        question: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        use_context: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a user query through the complete RAG pipeline.
        
        PIPELINE FLOW:
        1. Validate input
        2. Retrieve relevant context
        3. Generate answer
        4. Format and return response
        
        Args:
            question: User's question
            chat_history: Optional conversation history
            use_context: Whether to use retrieval (True) or just LLM (False)
            
        Returns:
            Response dict with answer and sources
        """
        logger.info("=" * 80)
        logger.info(f"QUERY: {question[:100]}")

        try:
            # Input validation
            validate_query(question)

            # Retrieve context
            context, retrieved_docs = self._retrieve_context(question)

            # Generate answer
            answer = self._generate_response(context, question)

            # Format response
            response = format_response(answer, retrieved_docs, question)

            logger.info("=" * 80)
            return response

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            return format_error_response(str(e), "validation_error")
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return format_error_response(str(e), "pipeline_error")
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

    def chat(self, question: str, chat_history: Optional[List[Tuple[str, str]]] = None) -> Dict[str, Any]:
        return self.query(question, chat_history)

    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        results = []
        for question in questions:
            results.append(self.query(question))
        return results


# Backwards-compatible alias used elsewhere in the repo
DrugRAGPipeline = IndianPharmaRAGPipeline
