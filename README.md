# 🏥 Indian Pharma RAG - Production-Ready AI System

A **Retrieval-Augmented Generation (RAG)** system for querying Indian pharmaceutical product data using semantic search and OpenAI's GPT models. Built for production deployment with FastAPI, Pinecone vector database, and OpenAI embeddings.

**Status**: ✅ Production-ready, fully deployed on Render/Railway

---

## 🎯 What is This System?

This is an **AI-powered drug information system** that answers questions about Indian pharmaceutical products. Instead of just searching keywords, it understands the **meaning** of your query and returns contextually relevant drug information.

### Example Queries It Can Answer:
- "What are side effects of Aspirin?"
- "Which antibiotic is manufactured by Cipla?"
- "What medicine should I take for fever?"
- "What's the cheapest paracetamol available?"
- "Tell me about drug interactions with Metformin"

---

## 🏗️ Architecture Overview

### How RAG Works (The Magic ✨)

```
USER QUESTION
    ↓
[EMBEDDING] → Convert to vector
    ↓
[SEMANTIC SEARCH] → Find similar drug documents in Pinecone
    ↓
[CONTEXT RETRIEVAL] → Get top-5 most relevant documents
    ↓
[PROMPT ENGINEERING] → Format context + question
    ↓
[LLM GENERATION] → GPT generates informed answer
    ↓
[SOURCE ATTRIBUTION] → Return answer with references
    ↓
FINAL ANSWER + SOURCES
```

### Key Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| **Embeddings** | Convert text to vectors for semantic search | OpenAI (text-embedding-3-small) |
| **Vector Database** | Store and search drug embeddings | Pinecone |
| **Semantic Search** | Find contextually relevant drugs | Cosine similarity in vector space |
| **LLM Generation** | Generate human-like answers | GPT-3.5-turbo or GPT-4 |
| **API Server** | REST endpoints for queries | FastAPI + Uvicorn |

---

## 📁 Project Structure

```
indian-drug-rag/
│
├── src/                          # Main source code
│   ├── __init__.py
│   ├── config.py                 # Configuration management (OpenAI, Pinecone, etc.)
│   ├── ingest.py                 # Data loading, cleaning, chunking, embedding
│   ├── retriever.py              # Semantic search and document retrieval
│   ├── rag_pipeline.py           # RAG pipeline: retrieve + generate
│   ├── api.py                    # FastAPI endpoints (production server)
│   └── utils.py                  # Helper functions, validation, formatting
│
├── data/
│   └── sample_drugs.csv          # Sample Indian drug dataset (20 drugs)
│
├── logs/                         # Application logs
│
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
├── README.md                     # This file
└── .gitignore                    # Git ignore rules
```

---

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Python 3.11+ (test with `python --version`)
- OpenAI API key (get from https://platform.openai.com/api-keys)
- Pinecone API key (free tier at https://www.pinecone.io)

### 1. Clone & Setup

```bash
# Clone repository
git clone <repo-url>
cd indian-drug-rag

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Or (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy template to actual .env
cp .env.example .env

# Edit .env and add your API keys
# OPENAI_API_KEY=sk-your-key
# PINECONE_API_KEY=pcak_your-key
```

### 3. Ingest Sample Data

```bash
# Load sample drugs into Pinecone
python src/ingest.py data/sample_drugs.csv
```

### 4. Start API Server

```bash
# Run FastAPI server
python -m uvicorn src.api:app --reload

# Server running at: http://localhost:8000
# Docs at: http://localhost:8000/docs
```

### 5. Test the System

**Option A: Using Swagger UI**
- Open http://localhost:8000/docs
- Click "Try it out" on `/query` endpoint
- Enter query: `"What is Aspirin used for?"`
- Click "Execute"

**Option B: Using cURL**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Aspirin used for?",
    "include_sources": true,
    "top_k": 5
  }'
```

**Option C: Using Python**
```python
import requests
import json

response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "What is Aspirin used for?",
        "include_sources": True,
        "top_k": 5
    }
)

result = response.json()
print(result["answer"])
print("Sources:", result["sources"])
```

---

## 🔍 Example Queries to Try

### Drug Information
- "Tell me about Crocin"
- "What is the price of Amoxicillin?"
- "Which manufacturer makes Metformin?"

### Medical Questions
- "What medicine can I take for fever?"
- "What are antibiotics available in India?"
- "Tell me about pain relievers"

### Side Effects & Safety
- "What are side effects of Aspirin?"
- "Is Paracetamol safe for children?"
- "Any contraindications with Losartan?"

### Manufacturer Queries
- "What drugs does Cipla manufacture?"
- "List all GlaxoSmithKline products"

---

## 🔧 File Explanations

### `config.py` - Configuration Management
**What it does**: Loads and validates all settings from environment variables

**Key configurations**:
```python
OPENAI_API_KEY              # Your OpenAI API key
EMBEDDING_MODEL             # Text embedding model
LLM_MODEL                  # LLM model for generation
PINECONE_API_KEY           # Pinecone vector DB key
CHUNK_SIZE                 # Characters per document chunk
TOP_K                      # Number of results to retrieve
```

### `ingest.py` - Data Ingestion Pipeline
**What it does**: Loads drug data, cleans it, generates embeddings, stores in Pinecone

**Process**:
1. **Load**: Read CSV/Excel/JSON files
2. **Clean**: Remove duplicates, handle missing values, standardize fields
3. **Chunk**: Split long documents into overlapping chunks
4. **Embed**: Generate vectors using OpenAI
5. **Upsert**: Store vectors in Pinecone with metadata

**Classes**:
- `IndianDrugDataProcessor`: Data loading and cleaning
- `EmbeddingGenerator`: Creates embeddings using OpenAI
- `PineconeUpserter`: Stores embeddings in Pinecone

### `retriever.py` - Semantic Search
**What it does**: Retrieves relevant drug documents from Pinecone

**Key features**:
- **Semantic search**: Finds similar documents by meaning, not keywords
- **Metadata filtering**: Filter by manufacturer, category, etc.
- **Relevance scoring**: Returns similarity scores (0-1)
- **Source attribution**: Tracks which documents were used

**Classes**:
- `SemanticRetriever`: Core semantic search functionality
- `HybridRetriever`: Combines semantic search with metadata filtering

### `rag_pipeline.py` - RAG Engine
**What it does**: Main RAG pipeline combining retrieval + LLM generation

**Workflow**:
1. Validate user query
2. Retrieve relevant context from Pinecone
3. Build prompt with context + history
4. Call GPT to generate answer
5. Format response with sources

**Key method**: `query(question)` - End-to-end query processing

### `api.py` - REST API Server
**What it does**: FastAPI server exposing RAG system as HTTP endpoints

**Endpoints**:

```
POST /query
├─ Input: { "query": "...", "include_sources": true, "top_k": 5 }
└─ Output: { "answer": "...", "sources": [...], "timestamp": "..." }

POST /ingest
├─ Input: { "filepath": "data/drugs.csv" }
└─ Output: { "success": true, "message": "..." }

GET /health
└─ Output: { "status": "healthy", "version": "1.0.0" }

GET /stats
└─ Output: { "uptime_seconds": 123.45, "queries_processed": 42 }
```

### `utils.py` - Helper Functions
**What it does**: Utility functions for validation, formatting, error handling

**Functions**:
- `validate_query()`: Check query is valid
- `validate_api_keys()`: Verify API keys configured
- `clean_text()`: Normalize text
- `extract_drug_name()`: Parse drug names from text
- `format_response()`: Format RAG output for API
- `retry_with_backoff()`: Decorator for retry logic
- `Timer`: Context manager for performance timing

---

## 💾 Adding New Drug Datasets

### Format: CSV File

Create a CSV with these columns:
```csv
drug_name,generic_name,manufacturer,category,dosage,indication,side_effects,price,composition
Aspirin,Acetylsalicylic Acid,Cipla Ltd,Pain Relief,500mg tablet,Pain relief,Stomach upset,₹5-15,Acetylsalicylic Acid 500mg
```

### Supported Columns (Optional)
- `drug_name` ⭐ (Required)
- `generic_name`
- `manufacturer`
- `category`
- `dosage`
- `indication` (uses)
- `side_effects`
- `price`
- `composition`
- Any custom fields!

### Ingest New Dataset

```bash
# Option 1: CLI
python src/ingest.py data/your_drugs.csv

# Option 2: API
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"filepath": "data/your_drugs.csv"}'

# Option 3: Python
from src.ingest import ingest_dataset
ingest_dataset("data/your_drugs.csv")
```

### Data Cleaning Process
The system automatically:
- ✅ Removes duplicates
- ✅ Handles missing values
- ✅ Standardizes column names
- ✅ Validates required fields
- ✅ Chunks text properly
- ✅ Generates embeddings
- ✅ Stores in Pinecone

---

## 🌍 Deployment Guide

### Local Development

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your keys

# 3. Run
python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

### Deploy to Render.com (FREE)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Create Render Service**
   - Go to https://render.com
   - Click "New" → "Web Service"
   - Connect GitHub repo
   - Configure:
     ```
     Build Command: pip install -r requirements.txt
     Start Command: uvicorn src.api:app --host 0.0.0.0 --port $PORT
     Environment Variables: (from .env file)
     ```
   - Deploy!

3. **Set Environment Variables**
   - In Render dashboard: Settings → Environment
   - Add: `OPENAI_API_KEY`, `PINECONE_API_KEY`, etc.

### Deploy to Railway.app (FREE)

1. **Connect GitHub**
   - Go to https://railway.app
   - Create new project
   - Select repository

2. **Configure**
   - Railway auto-detects Python
   - Add environment variables
   - Set start command: `uvicorn src.api:app --host 0.0.0.0 --port $PORT`

3. **Deploy**
   - Railway auto-deploys on git push!

### Deploy to Heroku (Free tier deprecated, but possible)

```bash
# Install Heroku CLI and login
heroku login

# Create app
heroku create your-app-name

# Set environment variables
heroku config:set OPENAI_API_KEY=your-key
heroku config:set PINECONE_API_KEY=your-key

# Deploy
git push heroku main
```

### Production Checklist

- [ ] Set `ENVIRONMENT=production` in .env
- [ ] Set `DEBUG=false`
- [ ] Use `gpt-3.5-turbo` (cheaper) or `gpt-4` (better)
- [ ] Configure proper logging
- [ ] Set up monitoring/alerts
- [ ] Use production Pinecone tier
- [ ] Enable HTTPS/SSL
- [ ] Configure CORS properly
- [ ] Set rate limits
- [ ] Backup your data

---

## 🔐 Security & Best Practices

### API Keys ⚠️
- **NEVER** commit `.env` to git
- Use `.env.example` as template
- Rotate keys periodically
- Use IAM roles in production

### Input Validation
```python
# All inputs validated before processing
- Query length: 3-500 characters
- File paths: Checked for existence
- API keys: Required validation
```

### Error Handling
```python
# Comprehensive error handling
- Rate limit retries with backoff
- API failure graceful degradation
- Detailed logging for debugging
```

### Rate Limiting
```python
RATE_LIMIT=60  # 60 requests per minute
```

---

## 📊 Performance Tips

### For Faster Responses
1. **Use smaller chunks** (CHUNK_SIZE=300)
2. **Reduce TOP_K** (TOP_K=3 instead of 5)
3. **Use gpt-3.5-turbo** (not gpt-4)
4. **Set LLM_TEMPERATURE=0.1** (deterministic)

### For Better Accuracy
1. **Increase TOP_K** (TOP_K=10)
2. **Use gpt-4** (more capable)
3. **Larger chunks** (CHUNK_SIZE=800)
4. **Use text-embedding-3-large** (more accurate)

### Cost Optimization
```python
# Current setup (cheapest)
EMBEDDING_MODEL = "text-embedding-3-small"  # ~$0.02 per 1M tokens
LLM_MODEL = "gpt-3.5-turbo"                 # ~$0.50 per 1M tokens
```

**Monthly estimate**: ~$5-10 for 1000 queries

---

## 🐛 Troubleshooting

### Issue: "OPENAI_API_KEY not found"
**Solution**: Create `.env` file with your OpenAI key
```bash
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-...
```

### Issue: "Pinecone connection failed"
**Solution**: Verify API key and network
```bash
# Test connection
python -c "from pinecone import Pinecone; Pinecone(api_key='your-key')"
```

### Issue: "No embeddings generated"
**Solution**: Check OpenAI API has credits
```bash
# Verify API key works
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Issue: "Slow response times"
**Solution**: Check Pinecone index status
- Fewer documents = faster
- Increase TOP_K gradually
- Use smaller CHUNK_SIZE

### Issue: "Out of context length"
**Solution**: Reduce LLM_MAX_TOKENS
```python
LLM_MAX_TOKENS = 512  # Smaller responses
```

---

## 📚 Learning Resources

### RAG Concepts
- [LLMs Explained](https://www.deeplearning.ai/short-courses/retrieval-augmented-generation-rag/)
- [Vector Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Semantic Search](https://www.pinecone.io/learn/vector-search-basics/)

### Our Technologies
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Pinecone Docs](https://docs.pinecone.io)
- [FastAPI Guide](https://fastapi.tiangolo.com)
- [Pydantic Docs](https://docs.pydantic.dev)

---

## 📞 Support & Contributing

### Report Issues
1. Check [Troubleshooting](#-troubleshooting) section
2. Check logs: `cat logs/app.log`
3. Create issue on GitHub

### Contributing
1. Fork repository
2. Create feature branch
3. Submit pull request

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🎓 Educational Value

This project demonstrates:
- **RAG Architecture**: Real-world implementation
- **Vector Databases**: Semantic search with Pinecone
- **LLM Integration**: OpenAI API usage
- **FastAPI**: Modern Python web framework
- **Production Deployment**: Cloud-ready code
- **Best Practices**: Error handling, logging, validation

---

## 🚀 What's Next?

### Future Enhancements
- [ ] Multi-language support (Hindi, Tamil, etc.)
- [ ] Conversation memory (multi-turn chat)
- [ ] Drug interaction checking
- [ ] Image processing (medicine packaging)
- [ ] Real-time drug pricing integration
- [ ] Mobile app (React Native)
- [ ] Advanced analytics dashboard

### Scaling Considerations
- Add Redis caching for popular queries
- Implement query rate limiting per user
- Add authentication/API keys
- Set up monitoring (Datadog, New Relic)
- Horizontal scaling with load balancer
- Async job queue (Celery) for ingestion

---

## 💡 Tips for Success

1. **Start Small**: Test with 20 drugs, then add more
2. **Monitor Costs**: OpenAI charges per token
3. **Iterate**: Try different chunk sizes, top_k values
4. **Test Thoroughly**: Use diverse queries
5. **Keep Logs**: Monitor logs/app.log for issues
6. **Update Data**: Regularly ingest new drug data

---

**Last Updated**: December 2024  
**Version**: 1.0.0  
**Status**: Production Ready ✅


3. **Configure environment:**
```bash
# Copy the example file
cp .env.example .env

# Edit .env with your credentials
# OPENAI_API_KEY=your_key_here
# PINECONE_API_KEY=your_key_here
```

4. **Update Pinecone settings (if needed):**
   - Check your Pinecone dashboard for the correct environment
   - Update `PINECONE_ENVIRONMENT` in `.env`

## Usage

### Run the RAG System

```bash
python -m src.main
```

This will:
1. Load your Indian medicine database (indian_medicine_data.csv)
2. Index it in Pinecone
3. Run demo queries
4. Enter chatbot mode where you can ask natural language questions

### Run the Web Chatbot

```bash
python -m src.web_app
```

Then open:

- `http://127.0.0.1:5000`

This will launch a web view with a chat interface that keeps your current session history.

### Chatbot Usage

When the application starts, type your question and press Enter. The system will keep conversation history during the session, so follow-up questions work naturally.

Example chat interactions:

- "What is Augmentin used for?"
- "Who manufactures Azithral?"
- "Tell me more about drugs containing Azithromycin."
- "What are the side effects of Allegra?"
- "How much does Ascoril LS Syrup cost?"

### Example Queries

- "What is Augmentin used for?"
- "Which antibiotics are available?"
- "Tell me about drugs containing Azithromycin"
- "What is the price of Allegra?"
- "Who manufactures Ascoril?"

### Running Tests

```bash
pytest tests/
```

## Configuration

Edit `.env` to customize:

- `OPENAI_API_KEY`: Your OpenAI API key
- `PINECONE_API_KEY`: Your Pinecone API key
- `EMBEDDING_MODEL`: Text embedding model (default: text-embedding-3-small)
- `LLM_MODEL`: Language model for responses (default: gpt-3.5-turbo)
- `CHUNK_SIZE`: Size of text chunks (default: 500)
- `TOP_K`: Number of documents to retrieve (default: 5)

## Adding Your Own Data

### Using CSV Format

1. Create a CSV file in the `data/` directory with columns:
   - `drug_name`
   - `generic_name`
   - `manufacturer`
   - `category`
   - `dosage`
   - `indication`
   - `side_effects`
   - `price`

2. Modify `src/main.py` to load your CSV:
```python
documents = loader.load_from_csv("data/your_drugs.csv")
```

### Using JSON Format

Similar process with JSON files:
```python
documents = loader.load_from_json("data/your_drugs.json")
```

### Using PDF Format

If you have drug information stored in a PDF, use the generic `PDFLoader`:
```python
from src.data_loader import PDFLoader

pdf_loader = PDFLoader()
documents = pdf_loader.load("data/your_drugs.pdf")
```

### Using the generic loaders directly

The repository also exposes loader helpers for structured datasets:
```python
from src.data_loader import CSVLoader, JSONLoader, PDFLoader

csv_loader = CSVLoader()
json_loader = JSONLoader()
pdf_loader = PDFLoader()

documents = csv_loader.load("data/your_drugs.csv")
documents = json_loader.load("data/your_drugs.json")
documents = pdf_loader.load("data/your_drugs.pdf")
```

## Architecture

### Data Flow

```
Raw Drug Data (CSV/JSON)
        ↓
  Data Loader
        ↓
  Text Embeddings (OpenAI)
        ↓
  Pinecone Vector Store
        ↓
  Retrieval (Vector Search)
        ↓
  RAG Pipeline (LLM)
        ↓
  User Response
```

### Components

1. **DataLoader**: Loads and formats drug data
2. **VectorStoreManager**: Manages Pinecone indices
3. **DrugRAGPipeline**: Orchestrates retrieval and generation
4. **Main**: Entry point with demo and interactive mode

## Logging

The system includes comprehensive logging. Check logs for:
- Document loading status
- Vector store operations
- Query processing

## Troubleshooting

### API Key Errors
- Verify `.env` file has correct credentials
- Check Pinecone environment matches your account

### Connection Issues
- Ensure internet connection is stable
- Verify Pinecone API key has necessary permissions

### Slow Queries
- Increase `TOP_K` for more context
- Check Pinecone index status in dashboard

## Future Enhancements

- [ ] Add more comprehensive Indian drug database
- [ ] Support for drug interactions checking
- [ ] Integration with Indian drug regulatory data
- [ ] Web interface (Flask/Streamlit)
- [ ] Multi-language support
- [ ] Dosage calculators
- [ ] Side effect interaction detection

## License

This project is open source and available for educational purposes.

## Support

For issues or questions, please check:
- [OpenAI Documentation](https://platform.openai.com/docs)
- [Pinecone Documentation](https://docs.pinecone.io)
- [LangChain Documentation](https://python.langchain.com)
