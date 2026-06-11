"""
COMPLETE IMPLEMENTATION GUIDE
Indian Pharma RAG System - Production Ready
============================================================================

This document provides a complete walkthrough of:
1. How each component works
2. How to run the system locally
3. How to deploy to production
4. How to add new drug datasets
5. Troubleshooting guide
"""

# ============================================================================
# PART 1: SYSTEM ARCHITECTURE EXPLAINED
# ============================================================================

RAG PIPELINE EXPLAINED:
========================

RAG = Retrieval-Augmented Generation

Instead of training an AI on all drug data (expensive & outdated),
RAG retrieves relevant documents at query time and gives them to GPT.

Step-by-step flow:

1. USER SUBMITS QUERY
   Input: "What are side effects of Aspirin?"
   
2. EMBEDDING GENERATION (OpenAI)
   - Query converted to 1536-dimensional vector
   - Captures semantic meaning of question
   - Example vector: [0.234, -0.512, 0.891, ...]
   
3. SEMANTIC SEARCH (Pinecone)
   - Finds drug documents with similar vectors
   - Uses cosine similarity (0-1 score)
   - Returns TOP-K (5) most relevant documents
   - Example results:
     * Aspirin (0.95 score)
     * Ibugesic (0.87 score)
     * Dolo 650 (0.82 score)
     * Crocin (0.78 score)
     * Paracetamol (0.75 score)
   
4. CONTEXT BUILDING
   - Formats retrieved documents into text
   - Includes metadata (manufacturer, dosage)
   - Adds conversation history if available
   - Creates comprehensive context string
   
5. PROMPT ENGINEERING
   Combines:
   - System prompt: "You are a pharmacist expert..."
   - Context: Retrieved drug documents
   - History: Previous conversation turns
   - Question: Current user query
   
6. LLM GENERATION (GPT)
   - GPT reads context + question
   - Generates natural language answer
   - Cites sources from context
   - Maintains medical accuracy
   
7. OUTPUT FORMATTING
   Response includes:
   - Answer: Generated response
   - Sources: Which drugs were referenced
   - Scores: Relevance scores
   - Timestamp: When response was generated


KEY ADVANTAGES:
================
✅ Always current (uses live database)
✅ Low cost (cheaper than fine-tuning)
✅ Transparent (shows sources)
✅ Customizable (easy to add new drugs)
✅ Accurate (leverages up-to-date data)


# ============================================================================
# PART 2: COMPONENT BREAKDOWN
# ============================================================================

1. CONFIG.PY - Configuration Manager
   =============================
   Purpose: Centralized configuration from environment
   
   Loads from .env:
   - OPENAI_API_KEY: For embeddings and LLM
   - EMBEDDING_MODEL: Which embedding model to use
   - LLM_MODEL: Which GPT model to use
   - PINECONE_API_KEY: Vector database access
   - CHUNK_SIZE: How to split documents
   - TOP_K: How many results to return
   
   Creates directories:
   - logs/: Application logs
   - data/: Drug datasets
   - models/: Cached models
   
   Config is injected into all other modules.
   Change it once, affects entire system!


2. INGEST.PY - Data Pipeline
   ==========================
   Purpose: Load → Clean → Embed → Store
   
   Classes:
   
   a) IndianDrugDataProcessor
      - load_csv(): Read CSV files
      - load_excel(): Read Excel files
      - load_json(): Read JSON files
      - clean_data(): Remove duplicates, handle missing values
      - chunk_drug_record(): Split into overlapping chunks
      - process_dataset(): Convert to embedding-ready format
      
      Chunking Strategy:
      - Larger chunks = more context but slower retrieval
      - Overlap ensures no info lost at boundaries
      - Example: 500 chars per chunk, 50 char overlap
      
   b) EmbeddingGenerator
      - generate_embeddings(): Batch embed texts
      - Uses OpenAI text-embedding-3-small (1536 dims)
      - Implements batching for efficiency
      - Caches results to avoid re-embedding
      
      Cost: ~$0.02 per 1M tokens
      Speed: ~100k embeddings per $1 spent
      
   c) PineconeUpserter
      - upsert_documents(): Store vectors in Pinecone
      - "Upsert" = Update if exists, Insert if new
      - Stores metadata with vectors
      - Handles batch operations efficiently
      
   Full Pipeline:
   ingest.py → data/drugs.csv
              → Load (pandas)
              → Clean (remove NaN, duplicates)
              → Chunk (split with overlap)
              → Embed (OpenAI)
              → Upsert (Pinecone)


3. RETRIEVER.PY - Semantic Search Engine
   =====================================
   Purpose: Find relevant drugs given a query
   
   Classes:
   
   a) SemanticRetriever
      Main search functionality:
      - search(): Universal semantic search
      - search_by_drug_name(): Exact name match
      - search_by_manufacturer(): Filter by company
      - search_by_category(): Filter by type
      
      Example Flow:
      Query: "What antibiotics does Cipla make?"
      
      1. Convert to embedding
      2. Search Pinecone for similar vectors
      3. Filter results by manufacturer="Cipla"
      4. Return top-5 with scores
      
      Result Format:
      {
        "id": "drug_5_chunk_0",
        "score": 0.92,  # Similarity 0-1
        "metadata": {
          "name": "Amoxicillin",
          "manufacturer": "Cipla Ltd",
          "dosage": "500mg capsule",
          "side_effects": "Allergic reactions",
          ...
        },
        "text": "Full drug information..."
      }
      
   b) HybridRetriever
      Two-stage retrieval:
      1. Semantic search on query
      2. Optional metadata filtering
      
      Use when you want both relevance AND specific filters


4. RAG_PIPELINE.PY - Core RAG Engine
   ==================================
   Purpose: Complete RAG workflow
   
   Class: IndianPharmaRAGPipeline
   
   Main Method: query(question)
   
   Workflow:
   1. Validate input (3-500 chars)
   2. Retrieve context (from Pinecone)
   3. Build prompt (context + history + question)
   4. Generate answer (with GPT)
   5. Format response (with sources)
   
   System Prompt:
   - Defines GPT behavior
   - "You are a pharmaceutical specialist"
   - Instructions: cite sources, mention side effects
   - Constraints: recommend doctors
   
   Temperature Setting:
   - 0.0 = deterministic (best for facts)
   - 0.5 = balanced
   - 1.0 = creative (not good for drugs!)
   - We use 0.3 (mostly factual)
   
   Error Handling:
   - Input validation errors → 400 response
   - API errors → 500 response
   - Rate limits → retry with backoff


5. API.PY - REST API Server
   ========================
   Purpose: Expose RAG as HTTP endpoints
   
   Framework: FastAPI (modern async framework)
   Server: Uvicorn (ASGI server)
   
   Endpoints:
   
   GET /health
   - Simple health check
   - Used by load balancers
   - Response: {"status": "healthy"}
   
   GET /stats
   - System statistics
   - Uptime, queries processed, errors
   - Response: {"uptime_seconds": 123.45, ...}
   
   POST /query
   - Main RAG endpoint
   - Input: {"query": "...", "top_k": 5}
   - Output: {"answer": "...", "sources": [...]}
   - This is the main endpoint!
   
   POST /ingest
   - Ingest new drug data
   - Input: {"filepath": "data/drugs.csv"}
   - Runs in background (async)
   - Returns immediately with status
   
   All endpoints have:
   - Request validation (Pydantic)
   - Error handling
   - Logging
   - Response formatting


6. UTILS.PY - Helper Functions
   ===========================
   Purpose: Common utilities used across modules
   
   Validation:
   - validate_query(): Check query is 3-500 chars
   - validate_api_keys(): Ensure API keys set
   
   Text Processing:
   - clean_text(): Remove special characters
   - extract_drug_name(): Parse drug name
   - extract_price(): Extract price from text
   
   Error Handling:
   - @retry_with_backoff: Decorator for retries
   - Handles transient API failures
   - Exponential backoff prevents overwhelming
   
   Formatting:
   - format_response(): Convert RAG output to API format
   - format_error_response(): Standard error format
   
   Performance:
   - Timer: Context manager for timing code
   - Logs execution time automatically


# ============================================================================
# PART 3: LOCAL SETUP & TESTING
# ============================================================================

STEP 1: PREREQUISITES
======================
Required:
- Python 3.11+ (check: python --version)
- OpenAI API key from https://platform.openai.com/api-keys
- Pinecone API key from https://www.pinecone.io (free tier)

Optional:
- Git for version control
- Postman/Insomnia for API testing
- VS Code for editing


STEP 2: CLONE & ENVIRONMENT
============================

# Clone repository
git clone https://github.com/your-user/indian-drug-rag.git
cd indian-drug-rag

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt


STEP 3: CONFIGURE
==================

# Copy template
cp .env.example .env

# Edit .env
# Add your API keys:
OPENAI_API_KEY=sk-... (from OpenAI)
PINECONE_API_KEY=pcak_... (from Pinecone)

# Test configuration
python -c "from src.config import *; print('Config loaded!')"


STEP 4: INGEST SAMPLE DATA
===========================

# Load sample drugs into Pinecone
python src/ingest.py data/sample_drugs.csv

# This will:
# 1. Load 20 sample Indian drugs
# 2. Clean and chunk the data
# 3. Generate embeddings (10-30 seconds)
# 4. Upload to Pinecone
# 5. Print confirmation messages


STEP 5: START API SERVER
=========================

# Run FastAPI server
python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# Output:
# INFO:     Uvicorn running on http://0.0.0.0:8000
# INFO:     Application startup complete


STEP 6: TEST THE SYSTEM
========================

Option A: Swagger UI (easiest)
- Open http://localhost:8000/docs
- Click on POST /query
- Click "Try it out"
- Enter: {"query": "What is Aspirin?"}
- Click "Execute"
- See response!

Option B: cURL
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Aspirin used for?", "top_k": 5}'

Option C: Python script
from src.rag_pipeline import IndianPharmaRAGPipeline
pipeline = IndianPharmaRAGPipeline()
response = pipeline.query("What is Aspirin?")
print(response["answer"])

Option D: Test file
python tests/test_pipeline.py


EXPECTED RESPONSE:
==================
{
  "query": "What is Aspirin?",
  "answer": "Aspirin is an acetylsalicylic acid manufactured by Cipla Ltd. 
It is used for pain relief and fever, available as 500mg tablets. 
Important side effects include stomach upset and allergic reactions. 
Price ranges from ₹5-15 per pack. Always consult a healthcare provider 
before use.",
  "sources": [
    {
      "id": "drug_0_chunk_0",
      "relevance_score": 0.95,
      "drug_name": "Aspirin",
      "manufacturer": "Cipla Ltd",
      "category": "Pain Relief"
    }
  ],
  "timestamp": "2024-12-09T10:30:45.123456"
}


# ============================================================================
# PART 4: ADDING NEW DRUG DATASETS
# ============================================================================

STEP 1: PREPARE CSV FILE
========================

Required columns:
- drug_name (required) ⭐

Optional columns:
- generic_name
- manufacturer
- category
- dosage
- indication (uses)
- side_effects
- price
- composition

Example:
drug_name,generic_name,manufacturer,category,dosage,indication,side_effects,price
Aspirin,Acetylsalicylic Acid,Cipla Ltd,Pain Relief,500mg,"Pain, fever","Upset stomach",₹5-15
Amoxicillin,Amoxicillin,Pfizer,Antibiotic,500mg,"Bacterial infections","Allergic reactions",₹20-40

Tips:
- Use consistent naming (e.g., always "Cipla Ltd", not "CIPLA" or "cipla")
- Include units (mg, mcg) in dosage
- Separate multiple items with commas in quotes: "Item 1, Item 2"
- Include Indian prices (₹ symbol optional)


STEP 2: INGEST DATA
===================

# Option A: CLI
python src/ingest.py data/your_drugs.csv

# Option B: API (if server running)
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"filepath": "data/your_drugs.csv"}'

# Option C: Python
from src.ingest import ingest_dataset
ingest_dataset("data/your_drugs.csv")


STEP 3: VERIFY INGESTION
=========================

# Check logs
tail -f logs/app.log

# Look for:
# - "Loaded X rows from CSV"
# - "Data cleaning complete. Final dataset: X drugs"
# - "Generated X documents from dataset"
# - "Upserting X documents to Pinecone..."
# - "All documents upserted successfully"

# Typical time: 10-30 seconds for 100 drugs


STEP 4: TEST QUERIES
====================

# Test with new drug data
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about [your drug name]"}'


# ============================================================================
# PART 5: DEPLOYMENT GUIDES
# ============================================================================

DEPLOYMENT OPTION 1: RENDER.COM
================================

Best for: Beginners, free tier available

1. Push to GitHub
   git add .
   git commit -m "Production ready"
   git push origin main

2. Create Render service
   - Go to https://render.com
   - Click "New" > "Web Service"
   - Connect GitHub repo
   - Select Branch: main

3. Configure Render
   Name: indian-pharma-rag
   Runtime: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: uvicorn src.api:app --host 0.0.0.0 --port $PORT
   
4. Add Environment Variables
   Go to Settings > Environment
   Add each from .env:
   - OPENAI_API_KEY
   - PINECONE_API_KEY
   - ENVIRONMENT=production
   - DEBUG=false

5. Deploy!
   Render auto-deploys on git push
   Your URL: https://indian-pharma-rag.onrender.com


DEPLOYMENT OPTION 2: RAILWAY.APP
==================================

Best for: Simple setup, auto-detect Python

1. Install Railway CLI
   npm install -g @railway/cli

2. Login & Create Project
   railway login
   railway init

3. Add Environment
   railway variables:set OPENAI_API_KEY=sk-...
   railway variables:set PINECONE_API_KEY=pcak_...

4. Deploy
   railway up

5. View Logs
   railway logs


DEPLOYMENT OPTION 3: HEROKU
============================

Best for: Popular option (though free tier removed)

1. Install Heroku CLI
   # Windows: chocolatey install heroku-cli
   # Mac: brew tap heroku/brew && brew install heroku

2. Login & Create
   heroku login
   heroku create your-app-name

3. Set Environment
   heroku config:set OPENAI_API_KEY=sk-...
   heroku config:set PINECONE_API_KEY=pcak_...

4. Create Procfile
   echo "web: uvicorn src.api:app --host 0.0.0.0 --port \$PORT" > Procfile

5. Deploy
   git push heroku main


PRODUCTION CHECKLIST
====================

[ ] Set ENVIRONMENT=production
[ ] Set DEBUG=false
[ ] Use gpt-3.5-turbo (cheaper) or gpt-4 (better)
[ ] Reduce CHUNK_SIZE for faster responses
[ ] Set up logging (check logs regularly)
[ ] Configure CORS properly
[ ] Add rate limiting
[ ] Monitor API usage & costs
[ ] Set up alerts for errors
[ ] Backup your Pinecone data
[ ] Use HTTPS/SSL
[ ] Rotate API keys regularly


# ============================================================================
# PART 6: TROUBLESHOOTING
# ============================================================================

ISSUE 1: OPENAI_API_KEY not found
==================================
Error: ValueError: OPENAI_API_KEY environment variable is required

Solution:
1. Check .env file exists in project root
2. Verify it contains: OPENAI_API_KEY=sk-...
3. Reload terminal/IDE
4. Test: echo $OPENAI_API_KEY (should print your key)


ISSUE 2: Pinecone connection failed
====================================
Error: PineconeException: Failed to connect to index

Solution:
1. Verify PINECONE_API_KEY in .env
2. Check Pinecone index name matches PINECONE_INDEX_NAME
3. Create index if it doesn't exist (ingest.py does this)
4. Test connection:
   from pinecone import Pinecone
   pc = Pinecone(api_key="your-key")
   print(pc.list_indexes())


ISSUE 3: No results returned from search
==========================================
Error: Retrieved 0 relevant documents

Solution:
1. Check data was ingested: Look for logs/app.log
2. Try simpler query: "What drugs do you have?"
3. Verify Pinecone index has data
4. Lower SIMILARITY_THRESHOLD in .env (e.g., 0.3)
5. Increase TOP_K (e.g., 10)


ISSUE 4: Slow response times
==============================
Symptom: Takes 10+ seconds to respond

Solution:
1. Reduce CHUNK_SIZE (e.g., 300)
2. Reduce TOP_K (e.g., 3)
3. Use gpt-3.5-turbo (faster than gpt-4)
4. Reduce LLM_MAX_TOKENS (e.g., 512)
5. Check network: ping api.openai.com


ISSUE 5: Out of memory
=======================
Error: MemoryError during embedding generation

Solution:
1. Reduce BATCH_SIZE (e.g., 8)
2. Process smaller CSV files
3. Increase virtual memory
4. Use instance with more RAM


ISSUE 6: Rate limit exceeded
==============================
Error: RateLimitError: 429 Too Many Requests

Solution:
1. Built-in retry with backoff handles this
2. Wait a few seconds, retry
3. Upgrade OpenAI plan for higher limits
4. Reduce rate in .env: RATE_LIMIT=30


ISSUE 7: CORS errors in frontend
==================================
Error: Access to XMLHttpRequest blocked by CORS policy

Solution:
Edit src/api.py CORS settings:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],  # Specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# PART 7: MONITORING & OPTIMIZATION
# ============================================================================

MONITORING
==========

Check logs:
tail -f logs/app.log

Key metrics to watch:
- Query processing time
- Embedding generation time
- Pinecone search latency
- API error rates
- Cost per request

Typical times:
- Query embedding: 100-300ms
- Pinecone search: 50-200ms
- LLM generation: 1-3 seconds
- Total: 2-5 seconds per query


COST OPTIMIZATION
==================

Current setup cost per query:
- Embedding: ~$0.00005
- LLM: ~$0.001
- Total: ~$0.0015 per query

For 1000 queries/month:
- Embedding cost: $0.05
- LLM cost: $1.00
- Total: ~$1.05

Ways to reduce cost:
1. Use smaller chunks (less to embed)
2. Use gpt-3.5-turbo (cheaper than gpt-4)
3. Reduce TOP_K (fewer documents to read)
4. Cache popular queries (in production)
5. Batch process where possible


PERFORMANCE OPTIMIZATION
=========================

For faster responses:
1. Reduce CHUNK_SIZE to 300
2. Set TOP_K to 3 (instead of 5)
3. Use gpt-3.5-turbo
4. Set LLM_TEMPERATURE to 0.1
5. Reduce LLM_MAX_TOKENS to 512

For better accuracy:
1. Increase CHUNK_SIZE to 800
2. Set TOP_K to 10
3. Use gpt-4
4. Use text-embedding-3-large
5. Add more drug data


# ============================================================================
# PART 8: ADVANCED USAGE
# ============================================================================

MULTI-TURN CONVERSATIONS
=========================

The system supports conversation history:

response = pipeline.query(
    "Tell me about Aspirin",
    chat_history=[
        ("What's a pain reliever?", "Aspirin is..."),
        ("What brand?", "Cipla makes...")
    ]
)

History is included in prompt for context.


CUSTOM METADATA FILTERING
==========================

Search by specific manufacturer:

retrieved = retriever.search_by_manufacturer(
    "antibiotics",
    manufacturer="Cipla Ltd",
    top_k=3
)

Or by category:

retrieved = retriever.search_by_category(
    "fever relief",
    category="Analgesic",
    top_k=5
)


BATCH PROCESSING
=================

Process multiple queries:

from src.utils import batch_items

queries = [
    "What is Aspirin?",
    "Tell me about Crocin",
    "List antibiotics"
]

for query in queries:
    response = pipeline.query(query)
    print(response["answer"])


# ============================================================================
# PART 9: GETTING HELP
# ============================================================================

Resources:
- OpenAI API Docs: https://platform.openai.com/docs
- Pinecone Docs: https://docs.pinecone.io
- FastAPI Guide: https://fastapi.tiangolo.com
- GitHub Issues: Create issue in repository

Debug Mode:
Change LOG_LEVEL=DEBUG in .env
Provides detailed logging of all operations


END OF GUIDE
============================================================================
"""
