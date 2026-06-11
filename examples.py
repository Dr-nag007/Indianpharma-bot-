"""
EXAMPLE USAGE SCRIPT
Demonstrates how to use the Indian Pharma RAG system

Run this after:
1. Setting up .env with API keys
2. Ingesting sample data (python src/ingest.py data/sample_drugs.csv)
3. Optionally starting the API server

This script shows:
- Direct RAG pipeline usage
- Retriever usage
- Error handling
- Different query types
"""

import sys
import os

# Add repository root to Python path for src package imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag_pipeline import IndianPharmaRAGPipeline
from src.retriever import SemanticRetriever, HybridRetriever
from src.utils import Timer, format_error_response
import json


def example_basic_query():
    """Example 1: Basic query to RAG pipeline."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: BASIC QUERY")
    print("=" * 60)

    try:
        pipeline = IndianPharmaRAGPipeline()

        query = "What is Aspirin used for?"
        print(f"\n🔍 Query: {query}")

        response = pipeline.query(query)

        print(f"\n💊 Answer:\n{response['answer']}")

        if response.get("sources"):
            print(f"\n📚 Sources ({len(response['sources'])} documents):")
            for i, source in enumerate(response["sources"], 1):
                print(
                    f"  {i}. {source.get('drug_name', 'Unknown')} "
                    f"(Relevance: {int(source['relevance_score'] * 100)}%)"
                )

    except Exception as e:
        print(f"❌ Error: {e}")


def example_medical_question():
    """Example 2: Medical question."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: MEDICAL QUESTION")
    print("=" * 60)

    try:
        pipeline = IndianPharmaRAGPipeline()

        query = "What medicine can I take for fever?"
        print(f"\n🔍 Query: {query}")

        response = pipeline.query(query)

        print(f"\n💊 Answer:\n{response['answer']}")

        print(f"\n📊 Response details:")
        print(f"  - Query: {response['query']}")
        print(f"  - Sources used: {len(response['sources'])}")
        print(f"  - Timestamp: {response['timestamp']}")

    except Exception as e:
        print(f"❌ Error: {e}")


def example_side_effects():
    """Example 3: Side effects question."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: SIDE EFFECTS QUERY")
    print("=" * 60)

    try:
        pipeline = IndianPharmaRAGPipeline()

        query = "What are the side effects of Metformin?"
        print(f"\n🔍 Query: {query}")

        with Timer("Query Processing"):
            response = pipeline.query(query)

        print(f"\n⚠️  Answer:\n{response['answer']}")

        if response.get("sources"):
            print(f"\n📋 Referenced drugs:")
            for source in response["sources"]:
                print(f"  - {source.get('drug_name')}: {source.get('category')}")

    except Exception as e:
        print(f"❌ Error: {e}")


def example_manufacturer_search():
    """Example 4: Search by manufacturer."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: SEMANTIC SEARCH BY MANUFACTURER")
    print("=" * 60)

    try:
        retriever = SemanticRetriever()

        query = "antibiotics"
        manufacturer = "Pfizer India"

        print(f"\n🔍 Searching for: {query}")
        print(f"   From: {manufacturer}")

        results = retriever.search_by_manufacturer(query, manufacturer)

        if results:
            print(f"\n✅ Found {len(results)} result(s):")
            for i, doc in enumerate(results, 1):
                print(f"\n  {i}. {doc['metadata'].get('name', 'Unknown')}")
                print(f"     Relevance: {int(doc['score'] * 100)}%")
                print(f"     Category: {doc['metadata'].get('category')}")
                print(f"     Price: {doc['metadata'].get('price')}")
        else:
            print("\n❌ No results found")

    except Exception as e:
        print(f"❌ Error: {e}")


def example_hybrid_retrieval():
    """Example 5: Hybrid retrieval with filtering."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: HYBRID RETRIEVAL WITH FILTERING")
    print("=" * 60)

    try:
        retriever = HybridRetriever()

        query = "pain relief"
        filters = {"category": "Analgesic"}

        print(f"\n🔍 Query: {query}")
        print(f"   Filters: {filters}")

        results = retriever.search_with_refinement(query, refine_by=filters)

        if results:
            print(f"\n✅ Found {len(results)} result(s):")
            for i, doc in enumerate(results, 1):
                print(f"\n  {i}. {doc['metadata'].get('name')}")
                print(f"     Manufacturer: {doc['metadata'].get('manufacturer')}")
                print(f"     Dosage: {doc['metadata'].get('dosage')}")
                print(f"     Relevance: {int(doc['score'] * 100)}%")
        else:
            print("\n❌ No results found with those filters")

    except Exception as e:
        print(f"❌ Error: {e}")


def example_direct_search():
    """Example 6: Direct semantic search."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: DIRECT SEMANTIC SEARCH")
    print("=" * 60)

    try:
        retriever = SemanticRetriever()

        queries = [
            "Paracetamol for fever",
            "Antibiotic for infection",
            "High blood pressure medication",
        ]

        for query in queries:
            print(f"\n🔍 Query: {query}")
            results = retriever.search(query, top_k=3)

            if results:
                for i, doc in enumerate(results, 1):
                    drug = doc["metadata"].get("name", "Unknown")
                    score = int(doc["score"] * 100)
                    print(f"  {i}. {drug} ({score}%)")
            else:
                print("  No results found")

    except Exception as e:
        print(f"❌ Error: {e}")


def example_error_handling():
    """Example 7: Error handling."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: ERROR HANDLING")
    print("=" * 60)

    try:
        pipeline = IndianPharmaRAGPipeline()

        # Test with invalid query (too short)
        print("\n❌ Testing with invalid query (too short):")
        response = pipeline.query("hi")  # Less than 3 chars

        if not response.get("success"):
            print(f"  Error: {response.get('error')}")
            print(f"  Type: {response.get('error_type')}")

    except Exception as e:
        # Expected to fail, but gracefully handled
        print(f"  ✓ Error caught gracefully: {type(e).__name__}")

    try:
        # Test with empty query
        print("\n❌ Testing with empty query:")
        response = pipeline.query("")

        if not response.get("success"):
            print(f"  Error: {response.get('error')}")

    except Exception as e:
        print(f"  ✓ Error caught gracefully: {type(e).__name__}")


def example_batch_queries():
    """Example 8: Batch processing multiple queries."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: BATCH QUERY PROCESSING")
    print("=" * 60)

    try:
        pipeline = IndianPharmaRAGPipeline()

        queries = [
            "What is Aspirin?",
            "Tell me about Crocin",
            "What medicine for fever?",
        ]

        print(f"\n Processing {len(queries)} queries...")

        results = []
        for i, query in enumerate(queries, 1):
            print(f"\n  {i}. {query}")
            response = pipeline.query(query)
            results.append(response)
            print(f"     ✓ Answer generated ({len(response['answer'])} chars)")

        print(f"\n✅ Batch processing complete!")
        print(f"   Total queries: {len(results)}")
        print(f"   Total sources: {sum(len(r.get('sources', [])) for r in results)}")

    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("🏥 INDIAN PHARMA RAG - USAGE EXAMPLES")
    print("=" * 60)

    examples = [
        ("Basic Query", example_basic_query),
        ("Medical Question", example_medical_question),
        ("Side Effects Query", example_side_effects),
        ("Manufacturer Search", example_manufacturer_search),
        ("Hybrid Retrieval", example_hybrid_retrieval),
        ("Direct Semantic Search", example_direct_search),
        ("Error Handling", example_error_handling),
        ("Batch Processing", example_batch_queries),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\n" + "-" * 60)
    print("Running all examples...\n")

    for name, func in examples:
        try:
            func()
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {e}")

    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Try the API: python -m uvicorn src.api:app --reload")
    print("2. Visit: http://localhost:8000/docs")
    print("3. Ingest more data: python src/ingest.py data/your_dataset.csv")
    print("\n")


if __name__ == "__main__":
    main()
