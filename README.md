# Lumora Analytics - Hybrid Enterprise RAG System

A production-grade **Hybrid Retrieval-Augmented Generation (RAG)** system combining structured and unstructured data with ML-powered routing.

## Features

- **Structured Query Engine**: Employee directory & Jira tickets using pandas
- **Semantic Search**: Vector embeddings + ChromaDB
- **Hybrid Retrieval**: BM25 + Dense + Cross-encoder reranking
- **ML Router**: Fine-tuned **DistilBERT** for query classification
- **Local LLM**: Ollama (llama3.2:1b) with citations
- **Beautiful UI**: Streamlit Chat Interface
- **SpaCy NER**: Robust name/entity extraction

## Tech Stack

- **Backend**: FastAPI + Streamlit
- **Vector DB**: ChromaDB
- **ML**: Hugging Face, DistilBERT (fine-tuned), SpaCy
- **LLM**: Ollama (llama3.2:1b)

## How to Run

1. Start Ollama:
   ```bash
   docker start ollama
   docker exec -it ollama ollama run llama3.2:1b