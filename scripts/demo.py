#!/usr/bin/env python3
"""
SupportMind Demo Script.
Demonstrates all system capabilities.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from supportmind.config.settings import get_config
from supportmind.stores.database import Database
from supportmind.stores.vector_store import VectorStore
from supportmind.pipelines.retrieval import UnifiedRetriever
from supportmind.pipelines.rag import RAGGenerator
from supportmind.pipelines.quality import QAScorer
from supportmind.pipelines.learning import LearningPipeline
from supportmind.analytics.metrics import Analytics
from supportmind.api.copilot import AgentCopilot
from supportmind.models.schemas import KnowledgeGap


def main():
    """Run the complete demo."""
    print("\n" + "=" * 70)
    print("🚀 SUPPORTMIND DEMO WALKTHROUGH")
    print("=" * 70)

    # Initialize components
    print("\nInitializing components...")
    config = get_config()
    db = Database()
    vs = VectorStore()

    # Try to load existing index
    vs.load()

    if vs.count() == 0:
        print("⚠️ Vector store is empty. Please run data ingestion first:")
        print("   python scripts/ingest_data.py --data-path ./data")
        return

    retriever = UnifiedRetriever(db, vs)
    rag_generator = RAGGenerator(retriever)
    qa_scorer = QAScorer(db)
    learning_pipeline = LearningPipeline(db, vs)
    analytics = Analytics(db, vs, retriever)
    copilot = AgentCopilot(rag_generator, qa_scorer, learning_pipeline)

    print(f"✅ Components initialized")
    print(f"   Vector store: {vs.count()} documents")
    print(f"   Device: {config.device}")

    # =========================================================================
    # DEMO 1: Basic RAG Query
    # =========================================================================
    print("\n" + "-" * 70)
    print("📝 DEMO 1: RAG Query & Response")
    print("-" * 70)

    test_query = "How do I reset a user's password in the system?"
    print(f"\n🔍 Query: {test_query}")

    response = rag_generator.generate(test_query)

    print(f"\n📄 Answer:\n{response.answer[:500]}...")
    print(f"\n📊 Metrics:")
    print(f"   - Detected Type: {response.detected_type}")
    print(f"   - Retrieval Confidence: {response.retrieval_confidence:.3f}")
    print(f"   - Score Margin: {response.score_margin:.3f}")
    print(f"   - Answer Confidence: {response.answer_confidence}")
    print(f"   - Processing Time: {response.processing_time_ms:.1f}ms")
    print(f"   - Trace ID: {response.trace_id}")

    print(f"\n📚 Sources ({len(response.sources)}):")
    for s in response.sources[:3]:
        print(f"   {s.rank}. {s.get_citation()} - {s.document.title[:50]}... (score: {s.score:.3f})")

    # =========================================================================
    # DEMO 2: QA Evaluation
    # =========================================================================
    print("\n" + "-" * 70)
    print("✅ DEMO 2: QA & Compliance Scoring")
    print("-" * 70)

    qa_result = qa_scorer.evaluate(
        question=test_query,
        response=response.answer,
        context="\n".join([s.document.title for s in response.sources]),
        trace_id=response.trace_id
    )

    print(f"\n📊 QA Scores:")
    print(f"   - Tone: {qa_result.tone_score:.1f}/100")
    print(f"   - Accuracy: {qa_result.accuracy_score:.1f}/100")
    print(f"   - Completeness: {qa_result.completeness_score:.1f}/100")
    print(f"   - Compliance: {qa_result.compliance_score:.1f}/100")
    print(f"   - Overall: {qa_result.overall_score:.1f}/100")
    print(f"   - Action: {qa_result.action.upper()}")
    print(f"   - Auto-Zero: {'❌ YES' if qa_result.auto_zero else '✅ No'}")

    if qa_result.suggestions:
        print(f"\n💡 Suggestions:")
        for s in qa_result.suggestions[:2]:
            print(f"   - {s}")

    # =========================================================================
    # DEMO 3: Agent Copilot
    # =========================================================================
    print("\n" + "-" * 70)
    print("🤖 DEMO 3: Agent Copilot")
    print("-" * 70)

    customer_msg = "I'm getting an error when trying to generate reports. Error code 5012."
    print(f"\n👤 Customer: {customer_msg}")

    suggestion = copilot.get_suggestion(customer_msg)

    print(f"\n🤖 Suggested Response:\n{suggestion['suggested_response'][:400]}...")
    print(f"\n📊 Copilot Metrics:")
    print(f"   - Confidence: {suggestion['confidence']}")
    print(f"   - Detected Type: {suggestion['detected_type']}")
    print(f"   - QA Preview: {suggestion['qa_preview']['overall_score']}/100 ({suggestion['qa_preview']['action']})")

    if suggestion['warnings']:
        print(f"\n⚠️ Warnings:")
        for w in suggestion['warnings']:
            print(f"   {w}")

    # =========================================================================
    # DEMO 4: Gap Detection
    # =========================================================================
    print("\n" + "-" * 70)
    print("🔬 DEMO 4: Knowledge Gap Detection")
    print("-" * 70)

    obscure_query = "How to configure advanced webhook integrations with custom headers?"
    obscure_response = rag_generator.generate(obscure_query)

    print(f"\n🔍 Query: {obscure_query}")
    print(f"   Retrieval Confidence: {obscure_response.retrieval_confidence:.3f}")
    print(f"   Score Margin: {obscure_response.score_margin:.3f}")

    gap = learning_pipeline.detect_gap(obscure_response)

    if gap:
        print(f"\n🚨 Gap Detected!")
        print(f"   - Gap ID: {gap.gap_id}")
        print(f"   - Type: {gap.gap_type}")
        print(f"   - Priority: {gap.priority}")
        print(f"   - Should Update Existing: {gap.should_update_existing}")
    else:
        print(f"\n✅ No significant gap detected")

    # =========================================================================
    # DEMO 5: Analytics
    # =========================================================================
    print("\n" + "-" * 70)
    print("📈 DEMO 5: Analytics & Metrics")
    print("-" * 70)

    # Retrieval accuracy
    print("\n🎯 Retrieval Accuracy (sample):")
    accuracy = analytics.evaluate_retrieval_accuracy(20)
    print(f"   - Hit@1: {accuracy['hit_at_1']*100:.1f}%")
    print(f"   - Hit@3: {accuracy['hit_at_3']*100:.1f}%")
    print(f"   - Hit@5: {accuracy['hit_at_5']*100:.1f}%")
    print(f"   - MRR: {accuracy['mrr']:.3f}")

    # Learning velocity
    print("\n📚 Learning Velocity (7 days):")
    velocity = analytics.get_learning_velocity(7)
    print(f"   - Gaps/day: {velocity['gaps_per_day']:.2f}")
    print(f"   - Drafts/day: {velocity['drafts_per_day']:.2f}")
    print(f"   - Published/day: {velocity['published_per_day']:.2f}")

    # Issue clusters
    print("\n🔍 Top Issue Clusters:")
    clusters = analytics.cluster_issues(2)
    for c in clusters[:3]:
        print(f"   - {c['category']}: {c['ticket_count']} tickets ({c['urgency']} urgency)")

    # KB stats
    print("\n📚 Knowledge Base Stats:")
    kb_stats = analytics.get_kb_freshness()
    print(f"   - Total KB Articles: {kb_stats['total_kb_articles']}")
    print(f"   - Auto-Generated: {kb_stats['auto_generated']}")

    # Vector store stats
    print("\n🗄️ Vector Store Stats:")
    print(f"   - Total Documents: {vs.count()}")
    print(f"   - By Type: {vs.counts_by_type()}")

    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
