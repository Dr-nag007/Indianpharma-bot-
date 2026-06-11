"""
Test script to verify Indian Pharma RAG system is working correctly

Run this after setup:
    python tests/test_setup.py

This will verify:
1. All dependencies installed
2. Configuration is valid
3. API keys are set
4. System can be imported
"""

import sys
import os

# Add repository root to path for src package imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from src import config
        print("  ✓ src.config")

        from src import ingest
        print("  ✓ src.ingest")

        from src import retriever
        print("  ✓ src.retriever")

        from src import rag_pipeline
        print("  ✓ src.rag_pipeline")

        from src import api
        print("  ✓ src.api")

        from src import utils
        print("  ✓ src.utils")

        print("✅ All imports successful!\n")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}\n")
        return False


def test_config():
    """Test configuration loading."""
    print("Testing configuration...")
    try:
        from src.config import (
            OPENAI_API_KEY,
            PINECONE_API_KEY,
            LLM_MODEL,
            EMBEDDING_MODEL,
        )

        # Check API keys
        if not OPENAI_API_KEY:
            print("  ⚠️  OPENAI_API_KEY not set in .env")
        else:
            key_preview = OPENAI_API_KEY[:10] + "***"
            print(f"  ✓ OPENAI_API_KEY: {key_preview}")

        if not PINECONE_API_KEY:
            print("  ⚠️  PINECONE_API_KEY not set in .env")
        else:
            key_preview = PINECONE_API_KEY[:10] + "***"
            print(f"  ✓ PINECONE_API_KEY: {key_preview}")

        print(f"  ✓ LLM_MODEL: {LLM_MODEL}")
        print(f"  ✓ EMBEDDING_MODEL: {EMBEDDING_MODEL}")
        print("✅ Configuration loaded!\n")
        return True

    except Exception as e:
        print(f"❌ Configuration failed: {e}\n")
        return False


def test_dependencies():
    """Test that all required packages are installed."""
    print("Testing dependencies...")
    required_packages = [
        ("openai", "OpenAI API SDK"),
        ("pinecone", "Pinecone vector database"),
        ("pandas", "Data manipulation"),
        ("fastapi", "Web framework"),
        ("uvicorn", "ASGI server"),
        ("pydantic", "Data validation"),
        ("dotenv", "Environment variables"),
    ]

    all_ok = True
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}: {description}")
        except ImportError:
            print(f"  ❌ {package}: Not installed")
            all_ok = False

    if all_ok:
        print("✅ All dependencies installed!\n")
    else:
        print("❌ Some dependencies missing. Run: pip install -r requirements.txt\n")

    return all_ok


def test_sample_data():
    """Test that sample data file exists."""
    print("Testing sample data...")
    sample_file = os.path.join(os.path.dirname(__file__), "..", "data", "sample_drugs.csv")

    if os.path.exists(sample_file):
        with open(sample_file, "r") as f:
            lines = f.readlines()
            num_drugs = len(lines) - 1  # Exclude header
            print(f"  ✓ Sample data file found ({num_drugs} drugs)")
        print("✅ Sample data ready!\n")
        return True
    else:
        print(f"  ❌ Sample data not found at {sample_file}\n")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("INDIAN PHARMA RAG - SYSTEM VERIFICATION")
    print("=" * 60)
    print()

    results = {
        "Imports": test_imports(),
        "Configuration": test_config(),
        "Dependencies": test_dependencies(),
        "Sample Data": test_sample_data(),
    }

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s}: {status}")

    print()

    all_passed = all(results.values())
    if all_passed:
        print("🎉 All tests passed! System is ready to use.")
        print()
        print("Next steps:")
        print("1. Add your API keys to .env (if not done yet)")
        print("2. Ingest sample data: python src/ingest.py data/sample_drugs.csv")
        print("3. Start server: python -m uvicorn src.api:app --reload")
        print("4. Open: http://localhost:8000/docs")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print()
        print("Common issues:")
        print("- Missing .env file: Copy .env.example to .env")
        print("- Missing API keys: Add OPENAI_API_KEY and PINECONE_API_KEY")
        print("- Missing dependencies: Run: pip install -r requirements.txt")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
