import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.rag import ask


def load_ground_truth(path):
    with open(path, 'r') as f:
        return json.load(f)


def score_answer(answer, expected):
    """
    Checks if the expected value appears in the answer.
    Strips formatting so $416,161 matches 416,161 matches 416161.
    """
    def normalize(text):
        return text.lower().replace(',', '').replace('$', '').replace('%', '').strip()

    return normalize(expected) in normalize(answer)


def run_evaluation(ground_truth_path):
    """
    Runs every question through the RAG pipeline and scores the results.
    Company filter is applied per question when specified in ground truth.
    """
    ground_truth = load_ground_truth(ground_truth_path)

    print("="*60)
    print("RAG PIPELINE EVALUATION")
    print("="*60)

    results = []
    correct = 0

    for i, item in enumerate(ground_truth):
        company = item.get("company")   # None for cross-company questions
        print(f"\nQuestion {i+1}/{len(ground_truth)}: {item['question']}")
        if company:
            print(f"Company filter: {company}")

        # Run through the pipeline with company filter
        result = ask(item['question'], company=company)
        answer = result['answer']

        # Score it
        passed = score_answer(answer, item['expected_answer'])
        if passed:
            correct += 1

        results.append({
            "question": item['question'],
            "company_filter": company,
            "expected": item['expected_answer'],
            "answer": answer,
            "passed": passed,
            "sources_retrieved": [
                f"{s['company']} | {s['source']} p{s['page_number']} ({s['similarity_score']})"
                for s in result['sources']
            ]
        })

        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"Status: {status}")
        print(f"Expected to contain: {item['expected_answer']}")
        print(f"Answer: {answer[:200]}...")

    # Summary
    score = correct / len(ground_truth) * 100
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Score: {correct}/{len(ground_truth)} ({score:.1f}%)")
    print("\nDetailed Results:")
    for r in results:
        status = "✓" if r['passed'] else "✗"
        company_label = f"[{r['company_filter']}]" if r['company_filter'] else "[All]"
        print(f"  {status} {company_label} {r['question']}")

    output = {
        "score": score,
        "correct": correct,
        "total": len(ground_truth),
        "results": results
    }

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"eval/evaluation_results_{timestamp}.json"

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nFull results saved to {output_path}")
    return output


if __name__ == "__main__":
    run_evaluation("eval/ground_truth.json")