"""
Tests for retrieval pipeline.
"""

import pytest
from supportmind.pipelines.retrieval import UnifiedRetriever


class TestUnifiedRetriever:
    """Test UnifiedRetriever class."""

    def test_retrieve_empty_store(self, db, vector_store):
        """Test retrieval with empty vector store."""
        retriever = UnifiedRetriever(db, vector_store)
        results = retriever.retrieve("test query")
        assert results == []

    def test_retrieve_with_documents(self, db, vector_store, sample_documents):
        """Test retrieval with documents."""
        # Add documents
        for doc in sample_documents:
            db.insert("documents", {
                'doc_id': doc.doc_id,
                'doc_type': doc.doc_type,
                'source_id': doc.source_id,
                'title': doc.title,
                'content': doc.content,
                'metadata': doc.metadata,
                'version': 1,
                'is_active': 1
            })

        vector_store.add_documents(sample_documents)

        retriever = UnifiedRetriever(db, vector_store)
        results = retriever.retrieve("password reset")

        assert len(results) > 0
        assert all(hasattr(r, 'score') for r in results)
        assert all(hasattr(r, 'document') for r in results)

    def test_detect_answer_type(self, db, vector_store, sample_documents):
        """Test answer type detection."""
        vector_store.add_documents(sample_documents)

        retriever = UnifiedRetriever(db, vector_store)

        # Mock results
        from supportmind.models.schemas import RetrievalResult
        results = [
            RetrievalResult(document=sample_documents[0], score=0.9, rank=1),
            RetrievalResult(document=sample_documents[1], score=0.8, rank=2),
        ]

        doc_type, confidence = retriever.detect_answer_type(results)

        assert doc_type == "KB"
        assert confidence > 0

    def test_score_margin(self, db, vector_store, sample_documents):
        """Test score margin calculation."""
        retriever = UnifiedRetriever(db, vector_store)

        from supportmind.models.schemas import RetrievalResult
        results = [
            RetrievalResult(document=sample_documents[0], score=0.9, rank=1),
            RetrievalResult(document=sample_documents[1], score=0.7, rank=2),
        ]

        margin = retriever.get_score_margin(results)

        assert margin == pytest.approx(0.2, rel=0.01)
