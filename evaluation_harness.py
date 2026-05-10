from src.pipeline.hybrid_pipeline import HybridPipelineFinal
import time

pipeline = HybridPipelineFinal()

TEST_CASES = [
    {"query": "List all members of the DevOps team", "expected_keywords": ["Aisha Patel", "Omar Shaikh", "Lucia Ferreira"]},
    {"query": "Who reports to Aisha Patel?", "expected_keywords": ["Omar Shaikh", "Lucia Ferreira", "Raj Iyer"]},
    {"query": "Who reports to Sarah Mitchell?", "expected_keywords": ["James Okafor", "Priya Nair"]},
    {"query": "Who manages Marcus Webb?", "expected_keywords": ["Sarah Mitchell"]},
    {"query": "What is our data retention policy?", "expected_keywords": ["Tier 1", "Tier 4", "TECH-42"]},
    {"query": "What security policies apply to StreamAPI?", "expected_keywords": ["mTLS", "SEC-ACCESS-001"]},
    {"query": "Employee details of Ryan Park", "expected_keywords": ["Ryan Park", "Software Engineer"]},
    {"query": "Show me all blocked tickets", "expected_keywords": ["TECH-"]},
    {"query": "What governance policies impact Snowflake usage?", "expected_keywords": ["Snowflake", "TECH-42"]},
]

def score_answer(answer: str, expected_keywords: list) -> float:
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return round(found / len(expected_keywords), 2)

def run_eval():
    print("="*90)
    print("🚀 LUMORA ANALYTICS - FINAL EVALUATION")
    print("="*90)
    
    scores = []
    for i, test in enumerate(TEST_CASES):
        print(f"\n[{i+1}/{len(TEST_CASES)}] {test['query']}")
        
        result = pipeline.handle_query(test['query'])
        answer = result.get("answer", "")
        
        score = score_answer(answer, test["expected_keywords"])
        scores.append(score)
        
        print(f"Score: {score:.2f} | Preview: {answer[:180]}...")
    
    avg = round(sum(scores) / len(scores) * 100, 1)
    print("\n" + "="*90)
    print(f"FINAL AVERAGE SCORE: {avg}%")
    print("="*90)

if __name__ == "__main__":
    run_eval()