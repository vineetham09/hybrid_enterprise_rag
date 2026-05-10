import requests
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

class ClassifyQuery:
    def __init__(self):
        self.model_path = Path("models/query_classifier")
        self.fine_tuned_model = None
        self.tokenizer = None
        self._load_fine_tuned_model()

    def _load_fine_tuned_model(self):
        """Load fine-tuned model if available"""
        try:
            if self.model_path.exists():
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.fine_tuned_model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                print("Loaded fine-tuned DistilBERT query classifier")
            else:
                print("Fine-tuned model not found. Using LLM fallback.")
        except Exception as e:
            print(f"Could not load fine-tuned model: {e}")

    def classify_query(self, query: str) -> str:
        # Try fine-tuned model first
        if self.fine_tuned_model and self.tokenizer:
            try:
                inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=128)
                with torch.no_grad():
                    outputs = self.fine_tuned_model(**inputs)
                    prediction = torch.argmax(outputs.logits, dim=1).item()
                
                labels = {0: "structured", 1: "semantic", 2: "hybrid"}
                return labels.get(prediction, "hybrid")
            except:
                pass  # fallback if inference fails

        # Fallback to Ollama LLM classifier
        return self._classify_with_llm(query)

    def _classify_with_llm(self, query: str) -> str:
        prompt = f"""You are a query classifier for an enterprise knowledge system.

Classify the query into EXACTLY one of these three categories:
- structured : asks about specific employees, ticket IDs, team members, managers, direct reports, ticket priority/status, or cross-team ticket relationships
- semantic   : asks about policies, documentation, architecture, definitions, or procedures
- hybrid     : requires BOTH employee/ticket data AND policy/documentation

Reply with ONE word only: structured, semantic, or hybrid.

Query: {query}
"""

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3.2:1b", "prompt": prompt, "stream": False},
                timeout=15
            )
            label = response.json()["response"].strip().lower()
            if label in ("structured", "semantic", "hybrid"):
                return label
        except:
            pass
        
        return "hybrid" 