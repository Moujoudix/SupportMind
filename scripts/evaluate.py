#!/usr/bin/env python3
"""
Evaluation Script.
Runs comprehensive evaluation of the system.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from supportmind.config.settings import get_config
from supportmind.stores.database import Database
from supportmind.stores.vector_store import VectorStore
from supportmind.pipelines.retrieval import UnifiedRetriever
from supportmind.pipelines.rag import RAGGenerator
from supportmind.analytics.metrics import Analytics


def evaluate_on_questions(sample_size: int = 100) -> pd.DataFrame:
    """
    Evaluate system on sample questions from Questions table.

    Args:
        sample_size: Number of questions to evaluate

    Returns:
        DataFrame with evaluation results
    """
    db = Database()
    vs = VectorStore()
    vs.load()

    retriever = UnifiedRetriever(db, vs)
    rag = RAGGenerator(retriever)

    questions = db.query(f"""
        SELECT Question_ID, Question_Text, Answer_Type, Target_ID,
               Target_Title, Difficulty, Product, Category
        FROM questions
        ORDER BY RANDOM()
        LIMIT {sample_size}
    """)

    results = []
    total = len(questions)

    print(f"\nEvaluating {total} questions...")

    for i, q in enumerate(questions):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{total}")

        response = rag.generate(q['Question_Text'])

        # Check if correct target was retrieved
        hit = False
        hit_rank = None
        for j, s in enumerate(response.sources):
            if s.document.source_id == q['Target_ID']:
                hit = True
                hit_rank = j + 1
                break

        results.append({
            'question_id': q['Question_ID'],
            'question': q['Question_Text'][:80] + "...",
            'expected_type': q['Answer_Type'],
            'detected_type': response.detected_type,
            'type_match': q['Answer_Type'] == response.detected_type,
            'target_id': q['Target_ID'],
            'hit': hit,
            'hit_rank': hit_rank,
            'confidence': round(response.retrieval_confidence, 3),
            'score_margin': round(response.score_margin, 3),
            'difficulty': q['Difficulty'],
            'product': q['Product'],
            'category': q['Category'],
            'processing_time_ms': round(response.processing_time_ms, 1)
        })

    df = pd.DataFrame(results)
    return df


def main():
    """Run evaluation."""
    print("=" * 60)
    print("SupportMind Evaluation")
    print("=" * 60)

    config = get_config()

    # Run evaluation
    df = evaluate_on_questions(100)

    # Calculate metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\n📊 Overall Metrics ({len(df)} questions):")
    print(f"   Hit@1: {df['hit'].mean() * 100:.1f}%")
    print(f"   Type Match Rate: {df['type_match'].mean() * 100:.1f}%")
    print(f"   Avg Confidence: {df['confidence'].mean():.3f}")
    print(f"   Avg Processing Time: {df['processing_time_ms'].mean():.1f}ms")

    # By difficulty
    print(f"\n📈 By Difficulty:")
    for diff in df['difficulty'].unique():
        subset = df[df['difficulty'] == diff]
        print(f"   {diff}: Hit@1={subset['hit'].mean()*100:.1f}% (n={len(subset)})")

    # By type
    print(f"\n📈 By Expected Type:")
    for t in df['expected_type'].unique():
        subset = df[df['expected_type'] == t]
        print(f"   {t}: Hit@1={subset['hit'].mean()*100:.1f}% (n={len(subset)})")

    # Save results
    output_dir = Path("artifacts/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save CSV
    csv_path = output_dir / f"evaluation_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to {csv_path}")

    # Save summary
    summary = {
        'timestamp': timestamp,
        'total_questions': len(df),
        'hit_at_1': df['hit'].mean(),
        'type_match_rate': df['type_match'].mean(),
        'avg_confidence': df['confidence'].mean(),
        'avg_processing_time_ms': df['processing_time_ms'].mean(),
        'by_difficulty': {
            diff: {
                'count': len(subset),
                'hit_at_1': subset['hit'].mean()
            }
            for diff, subset in df.groupby('difficulty')
        }
    }

    json_path = output_dir / f"summary_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary saved to {json_path}")


if __name__ == "__main__":
    main()
