from unstructured_ingest_v1 import UnstructuredIngestion
from pathlib import Path

if __name__ == "__main__":
    print("Starting ingestion...")
    ingestor = UnstructuredIngestion(
        kb_path="data/knowledge_base"
    )
    ingestor.ingest()