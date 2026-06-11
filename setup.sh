#!/bin/bash
# QUICK START SCRIPT FOR INDIAN PHARMA RAG
# Run this script to set up the entire system

echo "=================================="
echo "Indian Pharma RAG - Quick Setup"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | grep -oP '\d+\.\d+')
echo "Found Python $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv
echo "✓ Virtual environment created"

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt
echo "✓ Dependencies installed"

# Copy .env file
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Edit .env and add your API keys:"
    echo "   OPENAI_API_KEY=sk-..."
    echo "   PINECONE_API_KEY=pcak_..."
else
    echo "✓ .env file already exists"
fi

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Edit .env and add your API keys"
echo "2. Run: python src/ingest.py data/sample_drugs.csv"
echo "3. Run: python -m uvicorn src.api:app --reload"
echo "4. Open: http://localhost:8000/docs"
echo ""
