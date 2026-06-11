import json
from pathlib import Path

import pytest
from PyPDF2 import PdfWriter

from src.data_loader import CSVLoader, JSONLoader, PDFLoader, IndianDrugDataLoader


def test_load_sample_data():
    """Test loading sample data"""
    loader = IndianDrugDataLoader("data")
    documents = loader.load_sample_data()
    
    assert len(documents) > 0
    assert documents[0].page_content is not None
    assert documents[0].metadata is not None


def test_format_drug_record():
    """Test drug record formatting"""
    record = {
        "name": "Test Drug",
        "dosage": "500mg",
        "price": "₹100"
    }
    
    formatted = IndianDrugDataLoader._format_drug_record(record)
    assert "Test Drug" in formatted
    assert "500mg" in formatted


def test_csv_loader(tmp_path):
    sample_csv = tmp_path / "drugs.csv"
    sample_csv.write_text("name,price\nTestDrug,100\n")

    loader = CSVLoader()
    documents = loader.load(str(sample_csv))

    assert len(documents) == 1
    assert "TestDrug" in documents[0].page_content


def test_json_loader(tmp_path):
    sample_json = tmp_path / "drugs.json"
    sample_json.write_text(json.dumps([{"name": "TestDrug", "price": 100}]))

    loader = JSONLoader()
    documents = loader.load(str(sample_json))

    assert len(documents) == 1
    assert "TestDrug" in documents[0].page_content


def test_pdf_loader(tmp_path):
    sample_pdf = tmp_path / "drugs.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with open(sample_pdf, "wb") as f:
        writer.write(f)

    loader = PDFLoader()
    documents = loader.load(str(sample_pdf))

    assert len(documents) == 1
    assert documents[0].metadata["source"] == "pdf"
