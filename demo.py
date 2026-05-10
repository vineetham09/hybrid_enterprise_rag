from src.pipeline.hybrid_pipeline import HybridPipelineFinal

def demo():
    p = HybridPipelineFinal()
    
    test_queries = [
        "List all members of the DevOps team",
        "What is our data retention policy?",
        "What security policies apply to StreamAPI?",
        "Show me all blocked tickets",
        "Who reports to Sarah Mitchell?"
    ]
    
    print("XYZ Analytics Hybrid RAG Demo\n" + "="*60)
    
    for q in test_queries:
        print(f"\n Query: {q}")
        result = p.handle_query(q)
        print(result['answer'][:600])
        print("-" * 80)

if __name__ == "__main__":
    demo()