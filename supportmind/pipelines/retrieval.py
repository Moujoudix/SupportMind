"""
Unified Retrieval Pipeline.
Hybrid search combining FAISS semantic + SQLite FTS5 keyword search.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from supportmind.config.settings import get_config
from supportmind.models.schemas import Document, RetrievalResult
from supportmind.stores.database import Database
from supportmind.stores.vector_store import VectorStore


class UnifiedRetriever:
    """
    Hybrid retrieval with evidence-based classification.
    Combines semantic search (FAISS) with keyword search (FTS5).
    """

    def __init__(
        self,
        db: Database = None,
        vector_store: VectorStore = None
    ):
        """
        Initialize retriever.

        Args:
            db: Database instance
            vector_store: VectorStore instance
        """
        self.db = db or Database()
        self.vs = vector_store or VectorStore()
        self.doc_cache: Dict[str, Document] = {}
        self.config = get_config()

    def _get_document(self, doc_id: str) -> Optional[Document]:
        """Get document from cache or database."""
        if doc_id in self.doc_cache:
            return self.doc_cache[doc_id]

        row = self.db.get("documents", "doc_id", doc_id)
        if not row:
            return None

        doc = Document(
            doc_id=row['doc_id'],
            doc_type=row['doc_type'],
            source_id=row['source_id'],
            title=row['title'],
            content=row['content'],
            metadata=row.get('metadata', {}),
            version=row.get('version', 1),
            created_at=row.get('created_at', ''),
            updated_at=row.get('updated_at', '')
        )
        self.doc_cache[doc_id] = doc
        return doc

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        doc_type: str = None
    ) -> List[RetrievalResult]:
        """
        Hybrid retrieval combining semantic and keyword search.

        Args:
            query: Search query
            top_k: Number of results to return
            doc_type: Optional filter by document type

        Returns:
            List of RetrievalResult with ranked documents
        """
        top_k = top_k or self.config.top_k

        # 1. Semantic search (FAISS)
        semantic_results = self.vs.search(query, top_k=top_k * 2, doc_type=doc_type)

        # 2. Keyword search (FTS5)
        keyword_results = self.db.fts_search(query, doc_type=doc_type, limit=top_k * 2)

        # 3. Merge scores
        combined: Dict[str, Dict] = {}

        for doc_id, score in semantic_results:
            combined[doc_id] = {'semantic': score, 'keyword': 0.0}

        for row in keyword_results:
            doc_id = row['doc_id']
            bm25 = abs(row.get('bm25_score', 0))
            kw_score = 1.0 / (1.0 + bm25) if bm25 else 0.5

            if doc_id in combined:
                combined[doc_id]['keyword'] = kw_score
            else:
                combined[doc_id] = {'semantic': 0.0, 'keyword': kw_score}

        # 4. Compute hybrid scores
        results = []
        for doc_id, scores in combined.items():
            hybrid = (
                self.config.semantic_weight * scores['semantic'] +
                self.config.keyword_weight * scores['keyword']
            )

            doc = self._get_document(doc_id)
            if not doc:
                continue

            results.append(RetrievalResult(
                document=doc,
                score=hybrid,
                rank=0,
                retrieval_method='hybrid',
                semantic_score=scores['semantic'],
                keyword_score=scores['keyword']
            ))

        # 5. Sort and rank
        results.sort(key=lambda x: x.score, reverse=True)
        for i, r in enumerate(results[:top_k]):
            r.rank = i + 1

        return results[:top_k]

    def detect_answer_type(
        self,
        results: List[RetrievalResult]
    ) -> Tuple[str, float]:
        """
        Determine Answer_Type from retrieved evidence.

        Args:
            results: List of retrieval results

        Returns:
            Tuple of (detected_type, confidence)
        """
        if not results:
            return "UNKNOWN", 0.0

        type_scores = defaultdict(float)
        for r in results:
            type_scores[r.document.doc_type] += r.score

        if not type_scores:
            return "UNKNOWN", 0.0

        best_type = max(type_scores, key=type_scores.get)
        total = sum(type_scores.values())
        confidence = type_scores[best_type] / total if total > 0 else 0.0

        return best_type, confidence

    def get_confidence(self, results: List[RetrievalResult]) -> float:
        """Get top retrieval confidence score."""
        return results[0].score if results else 0.0

    def get_score_margin(self, results: List[RetrievalResult]) -> float:
        """
        Get margin between top-1 and top-2 scores.
        Used as signal for gap detection.
        """
        if len(results) < 2:
            return 1.0
        return results[0].score - results[1].score

    def clear_cache(self):
        """Clear document cache."""
        self.doc_cache.clear()
