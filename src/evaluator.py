"""
evaluator.py
------------
Runs a set of test questions through the RAG pipeline and measures:
- Source precision (did we retrieve from the correct document?)
- Keyword hit rate (does the answer contain expected keywords?)
Writes results to evaluation/eval_results.md
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any

from src.llm_chain import ask


EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation")
QUESTIONS_FILE = os.path.join(EVAL_DIR, "test_questions.json")
RESULTS_FILE = os.path.join(EVAL_DIR, "eval_results.md")


def load_test_questions() -> List[Dict[str, Any]]:
    """Load test questions from JSON file."""
    if not os.path.exists(QUESTIONS_FILE):
        raise FileNotFoundError(f"Test questions file not found: {QUESTIONS_FILE}")
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_single(question_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run one test question and evaluate the result.

    Returns:
        Dict with question, answer, sources_found, source_precision, keyword_hit_rate
    """
    question = question_data["question"]
    expected_sources = question_data.get("expected_sources", [])
    expected_keywords = question_data.get("expected_keywords", [])

    print(f"\n  Q: {question}")
    answer, chunks = ask(question, k=5)

    # Metric 1: Source Precision
    retrieved_sources = set(c["source"] for c in chunks)
    expected_set = set(expected_sources)
    if expected_set:
        hits = len(retrieved_sources & expected_set)
        source_precision = round(hits / len(expected_set) * 100, 1)
    else:
        source_precision = None   # no expectation set

    # Metric 2: Keyword Hit Rate
    answer_lower = answer.lower()
    if expected_keywords:
        keyword_hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
        keyword_hit_rate = round(keyword_hits / len(expected_keywords) * 100, 1)
    else:
        keyword_hit_rate = None

    return {
        "question": question,
        "answer": answer,
        "retrieved_sources": sorted(retrieved_sources),
        "expected_sources": expected_sources,
        "source_precision": source_precision,
        "keyword_hit_rate": keyword_hit_rate,
        "num_chunks": len(chunks),
        "top_scores": [c["score"] for c in chunks[:3]],
    }


def run_evaluation() -> None:
    """Run the full evaluation suite and write results to markdown."""
    print("\n[START] Starting Evaluation Pipeline...")
    print("=" * 60)

    questions = load_test_questions()
    results = []

    for i, q_data in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] Evaluating...")
        result = evaluate_single(q_data)
        results.append(result)
        print(f"  Source Precision: {result['source_precision']}%")
        print(f"  Keyword Hit Rate: {result['keyword_hit_rate']}%")

    # Calculate overall stats
    sp_scores = [r["source_precision"] for r in results if r["source_precision"] is not None]
    kh_scores = [r["keyword_hit_rate"] for r in results if r["keyword_hit_rate"] is not None]
    avg_sp = round(sum(sp_scores) / len(sp_scores), 1) if sp_scores else 0
    avg_kh = round(sum(kh_scores) / len(kh_scores), 1) if kh_scores else 0

    # Write markdown report
    write_results_markdown(results, avg_sp, avg_kh)
    print(f"\n[DONE] Evaluation complete!")
    print(f"   Avg Source Precision : {avg_sp}%")
    print(f"   Avg Keyword Hit Rate : {avg_kh}%")
    print(f"   Report saved to      : {RESULTS_FILE}")


def write_results_markdown(results, avg_sp, avg_kh) -> None:
    """Write evaluation results as a formatted markdown file."""
    os.makedirs(EVAL_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# RAG Pipeline Evaluation Results",
        f"\n**Run date:** {timestamp}  ",
        f"**Total questions:** {len(results)}  ",
        f"**Avg Source Precision:** {avg_sp}%  ",
        f"**Avg Keyword Hit Rate:** {avg_kh}%  \n",
        "---\n",
        "## Summary Table\n",
        "| # | Question | Source Precision | Keyword Hit Rate | Chunks Retrieved |",
        "|---|----------|-----------------|-----------------|-----------------|",
    ]

    for i, r in enumerate(results, 1):
        sp = f"{r['source_precision']}%" if r['source_precision'] is not None else "N/A"
        kh = f"{r['keyword_hit_rate']}%" if r['keyword_hit_rate'] is not None else "N/A"
        q_short = r['question'][:60] + "..." if len(r['question']) > 60 else r['question']
        lines.append(f"| {i} | {q_short} | {sp} | {kh} | {r['num_chunks']} |")

    lines.append("\n---\n")
    lines.append("## Detailed Results\n")

    for i, r in enumerate(results, 1):
        lines.append(f"### Question {i}")
        lines.append(f"\n**Q:** {r['question']}\n")
        lines.append(f"**A:** {r['answer']}\n")
        lines.append(f"**Retrieved from:** {', '.join(r['retrieved_sources']) or 'None'}")
        lines.append(f"**Expected from:** {', '.join(r['expected_sources']) or 'Any'}")
        lines.append(f"**Source Precision:** {r['source_precision']}%")
        lines.append(f"**Keyword Hit Rate:** {r['keyword_hit_rate']}%")
        lines.append(f"**Top chunk scores:** {r['top_scores']}\n")
        lines.append("---\n")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_evaluation()
