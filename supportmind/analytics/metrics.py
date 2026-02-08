"""
Analytics and Metrics Module.
"""

import hashlib
from collections import defaultdict
from typing import Any, Dict, List

from supportmind.stores.database import Database
from supportmind.stores.vector_store import VectorStore
from supportmind.pipelines.retrieval import UnifiedRetriever


class Analytics:
    """Analytics and metrics for SupportMind."""

    def __init__(
        self,
        db: Database = None,
        vector_store: VectorStore = None,
        retriever: UnifiedRetriever = None
    ):
        """
        Initialize analytics.

        Args:
            db: Database instance
            vector_store: VectorStore instance
            retriever: UnifiedRetriever instance
        """
        self.db = db or Database()
        self.vs = vector_store or VectorStore()
        self.retriever = retriever or UnifiedRetriever(self.db, self.vs)

    def evaluate_retrieval_accuracy(self, sample_size: int = 100) -> Dict[str, float]:
        """
        Evaluate retrieval accuracy using Questions table.

        Args:
            sample_size: Number of questions to evaluate

        Returns:
            Dictionary with hit@k and MRR metrics
        """
        questions = self.db.query(f"""
            SELECT Question_ID, Question_Text, Answer_Type, Target_ID, Target_Title
            FROM questions
            LIMIT {sample_size}
        """)

        if not questions:
            return {
                'hit_at_1': 0, 'hit_at_3': 0, 'hit_at_5': 0,
                'mrr': 0, 'total_evaluated': 0
            }

        hits_at_1 = 0
        hits_at_3 = 0
        hits_at_5 = 0
        reciprocal_ranks = []

        for q in questions:
            query = q['Question_Text']
            expected_id = q['Target_ID']

            # Retrieve
            results = self.retriever.retrieve(query, top_k=5)

            # Check hits
            found_rank = None
            for i, r in enumerate(results):
                if r.document.source_id == expected_id:
                    found_rank = i + 1
                    break

            if found_rank:
                if found_rank <= 1:
                    hits_at_1 += 1
                if found_rank <= 3:
                    hits_at_3 += 1
                if found_rank <= 5:
                    hits_at_5 += 1
                reciprocal_ranks.append(1.0 / found_rank)
            else:
                reciprocal_ranks.append(0.0)

        n = len(questions)
        return {
            'hit_at_1': hits_at_1 / n if n > 0 else 0,
            'hit_at_3': hits_at_3 / n if n > 0 else 0,
            'hit_at_5': hits_at_5 / n if n > 0 else 0,
            'mrr': sum(reciprocal_ranks) / n if n > 0 else 0,
            'total_evaluated': n
        }

    def get_qa_trends(self, days: int = 30) -> List[Dict]:
        """
        Get QA score trends over time.

        Args:
            days: Number of days to look back

        Returns:
            List of daily QA statistics
        """
        try:
            return self.db.query(f"""
                SELECT
                    date(evaluated_at) as date,
                    COUNT(*) as evaluations,
                    AVG(overall_score) as avg_score,
                    AVG(tone_score) as avg_tone,
                    AVG(accuracy_score) as avg_accuracy,
                    AVG(completeness_score) as avg_completeness,
                    AVG(compliance_score) as avg_compliance,
                    SUM(CASE WHEN auto_zero = 1 THEN 1 ELSE 0 END) as auto_zero_count
                FROM qa_scores
                WHERE evaluated_at >= date('now', '-{days} days')
                GROUP BY date(evaluated_at)
                ORDER BY date
            """)
        except Exception as e:
            print(f"⚠️ QA trends query failed: {e}")
            return []

    def get_kb_freshness(self) -> Dict[str, Any]:
        """
        Analyze knowledge base freshness.

        Returns:
            Dictionary with KB statistics
        """
        stats = {}

        # Total KB articles
        try:
            total = self.db.query(
                "SELECT COUNT(*) as c FROM documents WHERE doc_type = 'KB'"
            )
            stats['total_kb_articles'] = total[0]['c'] if total else 0
        except Exception:
            stats['total_kb_articles'] = 0

        # By source type
        try:
            by_source = self.db.query("""
                SELECT
                    json_extract(metadata, '$.source_type') as source_type,
                    COUNT(*) as count
                FROM documents
                WHERE doc_type = 'KB'
                GROUP BY source_type
            """)
            stats['by_source'] = {
                r['source_type']: r['count']
                for r in by_source if r['source_type']
            }
        except Exception:
            stats['by_source'] = {}

        # Auto-generated count
        try:
            auto_gen = self.db.query("""
                SELECT COUNT(*) as c FROM documents
                WHERE doc_type = 'KB'
                AND json_extract(metadata, '$.source_type') = 'auto_generated'
            """)
            stats['auto_generated'] = auto_gen[0]['c'] if auto_gen else 0
        except Exception:
            stats['auto_generated'] = 0

        return stats

    def cluster_issues(self, min_cluster_size: int = 3) -> List[Dict]:
        """
        Basic root cause mining by clustering similar tickets.

        Args:
            min_cluster_size: Minimum tickets to form a cluster

        Returns:
            List of issue clusters
        """
        try:
            tickets = self.db.query("""
                SELECT Ticket_Number, Category, Product, Module, Root_Cause, Subject
                FROM tickets
                WHERE Root_Cause IS NOT NULL AND Root_Cause != ''
            """)

            if not tickets:
                return []

            # Group by category and similar root cause
            clusters = defaultdict(list)
            for t in tickets:
                root_cause = t.get('Root_Cause', '') or ''
                category = t.get('Category', '') or 'Unknown'
                key = f"{category}|{root_cause[:50]}"
                clusters[key].append(t)

            # Filter by minimum size and format
            result = []
            for key, ticket_list in clusters.items():
                if len(ticket_list) >= min_cluster_size:
                    parts = key.split('|', 1)
                    category = parts[0] if len(parts) > 0 else 'Unknown'
                    result.append({
                        'cluster_id': hashlib.md5(key.encode()).hexdigest()[:8],
                        'category': category,
                        'common_root_cause': ticket_list[0].get('Root_Cause', '')[:200],
                        'ticket_count': len(ticket_list),
                        'ticket_numbers': [t['Ticket_Number'] for t in ticket_list[:10]],
                        'affected_products': list(set(
                            t.get('Product', '') for t in ticket_list if t.get('Product')
                        )),
                        'urgency': 'high' if len(ticket_list) >= 10 else 'medium'
                    })

            result.sort(key=lambda x: x['ticket_count'], reverse=True)
            return result[:20]

        except Exception as e:
            print(f"⚠️ Cluster issues failed: {e}")
            return []

    def get_learning_velocity(self, days: int = 30) -> Dict[str, float]:
        """
        Calculate learning velocity metrics.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with learning metrics
        """
        # Gaps detected
        try:
            gaps = self.db.query(f"""
                SELECT COUNT(*) as c FROM knowledge_gaps
                WHERE detected_at >= date('now', '-{days} days')
            """)
            gaps_count = gaps[0]['c'] if gaps else 0
        except Exception:
            gaps_count = 0

        # Drafts created
        try:
            drafts = self.db.query(f"""
                SELECT COUNT(*) as c FROM draft_kb_articles
                WHERE created_at >= date('now', '-{days} days')
            """)
            drafts_count = drafts[0]['c'] if drafts else 0
        except Exception:
            drafts_count = 0

        # Published
        try:
            published = self.db.query("""
                SELECT COUNT(*) as c FROM draft_kb_articles
                WHERE status = 'approved'
            """)
            published_count = published[0]['c'] if published else 0
        except Exception:
            published_count = 0

        return {
            'gaps_per_day': gaps_count / days if days > 0 else 0,
            'drafts_per_day': drafts_count / days if days > 0 else 0,
            'published_per_day': published_count / days if days > 0 else 0,
            'conversion_rate': published_count / drafts_count if drafts_count > 0 else 0,
            'total_gaps': gaps_count,
            'total_drafts': drafts_count,
            'total_published': published_count
        }

    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get all data needed for dashboard.

        Returns:
            Comprehensive dashboard data dictionary
        """
        from supportmind.pipelines.learning import LearningPipeline

        learning_pipeline = LearningPipeline(self.db, self.vs)

        return {
            'retrieval_accuracy': self.evaluate_retrieval_accuracy(50),
            'qa_trends': self.get_qa_trends(7),
            'kb_freshness': self.get_kb_freshness(),
            'issue_clusters': self.cluster_issues(2),
            'learning_velocity': self.get_learning_velocity(7),
            'vector_store_stats': {
                'total_documents': self.vs.count(),
                'by_type': self.vs.counts_by_type()
            },
            'learning_stats': learning_pipeline.get_learning_stats()
        }

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health metrics.

        Returns:
            System health summary
        """
        health = {
            'status': 'healthy',
            'issues': []
        }

        # Check vector store
        doc_count = self.vs.count()
        if doc_count == 0:
            health['status'] = 'unhealthy'
            health['issues'].append('Vector store is empty')

        # Check database tables
        tables = ['documents', 'tickets', 'questions']
        for table in tables:
            try:
                result = self.db.query(f"SELECT COUNT(*) as c FROM {table}")
                if result[0]['c'] == 0:
                    health['issues'].append(f'Table {table} is empty')
            except Exception as e:
                health['status'] = 'unhealthy'
                health['issues'].append(f'Cannot access table {table}: {e}')

        if health['issues'] and health['status'] == 'healthy':
            health['status'] = 'degraded'

        health['document_count'] = doc_count
        health['document_types'] = self.vs.counts_by_type()

        return health
