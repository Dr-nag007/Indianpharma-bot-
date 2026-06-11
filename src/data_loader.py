import logging
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

try:
    from PyPDF2 import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None


class BaseDocumentLoader:
    """Base loader for structured drug data."""

    def _documents_from_dataframe(self, df: pd.DataFrame, source: str) -> List[Document]:
        documents = []
        for idx, row in df.iterrows():
            record = row.to_dict()
            content = self._format_drug_record(record)
            metadata = {
                "source": source,
                "row_index": idx,
                **{k: str(v) for k, v in record.items()}
            }
            documents.append(Document(page_content=content, metadata=metadata))

        logger.info(f"Loaded {len(documents)} items from {source} source")
        return documents

    @staticmethod
    def _format_drug_record(record: Dict[str, Any]) -> str:
        """Format a drug record for indexing."""
        content_parts = []

        priority_keys = [
            "name",
            "price(₹)",
            "manufacturer_name",
            "type",
            "pack_size_label",
            "short_composition1",
            "short_composition2",
            "drug_name",
            "generic_name",
            "manufacturer",
            "category",
            "dosage",
            "indication",
            "side_effects",
            "price",
        ]

        for key in priority_keys:
            if key in record and pd.notna(record[key]):
                content_parts.append(f"{key}: {record[key]}")

        for key, value in record.items():
            if key not in priority_keys and pd.notna(value):
                content_parts.append(f"{key}: {value}")

        return "\n".join(content_parts)


class CSVLoader(BaseDocumentLoader):
    """Generic CSV loader."""

    def load(self, file_path: str) -> List[Document]:
        file = Path(file_path)
        if not file.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        try:
            df = pd.read_csv(file_path)
            return self._documents_from_dataframe(df, source="csv")
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise


class JSONLoader(BaseDocumentLoader):
    """Generic JSON loader."""

    def load(self, file_path: str) -> List[Document]:
        file = Path(file_path)
        if not file.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        try:
            df = pd.read_json(file_path)
            return self._documents_from_dataframe(df, source="json")
        except Exception as e:
            logger.error(f"Error loading JSON file: {e}")
            raise


class PDFLoader(BaseDocumentLoader):
    """Generic PDF loader."""

    def load(self, file_path: str) -> List[Document]:
        if PdfReader is None:
            raise ImportError("PyPDF2 is required for PDFLoader. Install it with 'pip install PyPDF2'.")

        file = Path(file_path)
        if not file.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        try:
            reader = PdfReader(str(file))
            documents = []
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                metadata = {
                    "source": "pdf",
                    "file_path": str(file),
                    "page_number": page_number,
                }
                documents.append(Document(page_content=text, metadata=metadata))

            logger.info(f"Loaded {len(documents)} pages from PDF file")
            return documents
        except Exception as e:
            logger.error(f"Error loading PDF file: {e}")
            raise


class IndianDrugDataLoader(BaseDocumentLoader):
    """Loader for Indian drug database"""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_from_csv(self, file_path: str) -> List[Document]:
        """Load drug data from a CSV file."""
        return CSVLoader().load(file_path)

    def load_from_json(self, file_path: str) -> List[Document]:
        """Load drug data from a JSON file."""
        return JSONLoader().load(file_path)

    def _documents_from_dataframe(self, df: pd.DataFrame, source: str) -> List[Document]:
        return super()._documents_from_dataframe(df, source)

    def load_sample_data(self) -> List[Document]:
        """Load sample Indian drug database."""
        sample_drugs = [
            {
                "drug_name": "Aspirin",
                "generic_name": "Acetylsalicylic Acid",
                "manufacturer": "Cipla Ltd",
                "category": "Pain Relief",
                "dosage": "500mg tablet",
                "indication": "Pain, fever, inflammation",
                "side_effects": "Stomach upset, allergic reactions",
                "price": "₹5-15",
            },
            {
                "drug_name": "Crocin",
                "generic_name": "Paracetamol",
                "manufacturer": "GlaxoSmithKline",
                "category": "Analgesic",
                "dosage": "500mg tablet",
                "indication": "Fever, headache, body pain",
                "side_effects": "Rare allergic reactions",
                "price": "₹10-20",
            },
            {
                "drug_name": "Amoxicillin",
                "generic_name": "Amoxicillin",
                "manufacturer": "Pfizer India",
                "category": "Antibiotic",
                "dosage": "500mg capsule",
                "indication": "Bacterial infections",
                "side_effects": "Diarrhea, rash, allergic reactions",
                "price": "₹20-50",
            },
            {
                "drug_name": "Metformin",
                "generic_name": "Metformin Hydrochloride",
                "manufacturer": "Sun Pharmaceutical",
                "category": "Antidiabetic",
                "dosage": "500mg tablet",
                "indication": "Type 2 diabetes",
                "side_effects": "Nausea, diarrhea, lactic acidosis (rare)",
                "price": "₹30-80",
            },
            {
                "drug_name": "Atorvastatin",
                "generic_name": "Atorvastatin Calcium",
                "manufacturer": "Ranbaxy",
                "category": "Antilipidemic",
                "dosage": "10mg tablet",
                "indication": "High cholesterol, cardiovascular disease",
                "side_effects": "Muscle pain, liver problems (rare)",
                "price": "₹50-150",
            },
        ]

        documents = []
        for idx, drug in enumerate(sample_drugs):
            content = self._format_drug_record(drug)
            metadata = {"source": "sample", "drug_id": idx, **drug}
            documents.append(Document(page_content=content, metadata=metadata))

        logger.info(f"Loaded {len(documents)} sample Indian drugs")
        return documents
