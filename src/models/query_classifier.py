from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

class QueryIntentClassifier:

    def __init__(self, model_name="distilbert-base-uncased", output_dir="models/query_classifier"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None

    def prepare_training_data(self):
        """Training data for query classification"""
        data = {
            "query": [
                # Structured
                "Who reports to James Okafor?", "List all members of the DevOps team",
                "Show me all blocked tickets", "Tell me about ticket TECH-42",
                "Who are the direct reports of Sarah Mitchell?", "List critical tickets",
                "Show employees in Engineering team", "Who manages Laura Hensley?",

                # Semantic
                "What is our remote work policy?", "Explain the data governance policy",
                "What does the security policy say about encryption?", 
                "Summarize the engineering architecture decisions",
                "What metrics are defined for InsightDash?",

                # Hybrid
                "What security policies apply to StreamAPI?", 
                "Which governance rules impact DataLake Pro?",
                "What tickets relate to cross-team initiatives?",
                "Show me blocked tickets related to Snowflake governance",
            ],
            "label": [0,0,0,0,0,0,0,0, 1,1,1,1,1, 2,2,2,2]
        }
        
        df = pd.DataFrame(data)
        return Dataset.from_pandas(df)

    def train(self):
        dataset = self.prepare_training_data()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        def tokenize_function(examples):
            return self.tokenizer(examples["query"], padding="max_length", truncation=True, max_length=128)

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=3
        )

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=6,
            per_device_train_batch_size=8,
            save_steps=50,
            logging_steps=10,
            eval_strategy="no",           # Fixed: changed from evaluation_strategy
            load_best_model_at_end=False,
            push_to_hub=False,
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )

        print("Starting fine-tuning of DistilBERT for query intent classification...")
        trainer.train()
        
        # Save model
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Fine-tuning completed successfully!")
        print(f"Model saved to: {self.output_dir}")


if __name__ == "__main__":
    classifier = QueryIntentClassifier()
    classifier.train()