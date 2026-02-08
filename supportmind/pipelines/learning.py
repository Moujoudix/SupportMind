"""
Self-Learning Pipeline.
Gap detection, KB generation, lineage tracking, and review workflow.
"""

import hashlib
import json
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from supportmind.config.settings import get_config, GapType
from supportmind.models.schemas import (
    Document,
    DraftKBArticle,
    KnowledgeGap,
    RAGResponse,
    QAScore,
)
from supportmind.stores.database import Database
from supportmind.stores.vector_store import VectorStore


class LearningPipeline:
    """
    Self-learning pipeline:
    1. Detect knowledge gaps
    2. Generate draft KB articles
    3. Track lineage/provenance
    4. Manage review workflow
    5. Publish and update index
    """

    def __init__(
        self,
        db: Database = None,
        vector_store: VectorStore = None
    ):
        """
        Initialize learning pipeline.

        Args:
            db: Database instance
            vector_store: VectorStore instance
        """
        self.db = db or Database()
        self.vs = vector_store or VectorStore()
        self.config = get_config()
        self.issue_tracker: Dict[str, int] = defaultdict(int)

    def detect_gap(
        self,
        rag_response: RAGResponse,
        qa_score: QAScore = None,
        resolution: str = ""
    ) -> Optional[KnowledgeGap]:
        """
        Detect knowledge gaps using multiple signals.

        Signals:
        1. Low retrieval confidence
        2. Small score margin (weak differentiation)
        3. QA auto-zero triggered
        4. Repeated similar issues

        Args:
            rag_response: RAG response to analyze
            qa_score: Optional QA score
            resolution: Optional resolution text

        Returns:
            KnowledgeGap if detected, None otherwise
        """
        signals = {
            'low_confidence': False,
            'small_margin': False,
            'auto_zero': False,
            'repeated': False
        }

        # Signal 1: Low retrieval confidence
        if rag_response.retrieval_confidence < self.config.gap_confidence_threshold:
            signals['low_confidence'] = True

        # Signal 2: Small score margin
        if rag_response.score_margin < self.config.score_margin_threshold:
            signals['small_margin'] = True

        # Signal 3: QA auto-zero
        if qa_score and qa_score.auto_zero:
            signals['auto_zero'] = True

        # Signal 4: Repeated similar issues
        query_hash = hashlib.md5(
            rag_response.query[:100].lower().encode()
        ).hexdigest()[:8]
        self.issue_tracker[query_hash] += 1
        if self.issue_tracker[query_hash] >= 3:
            signals['repeated'] = True

        # Determine if gap exists
        signal_count = sum(signals.values())
        has_gap = signal_count >= 2 or signals['auto_zero']

        if not has_gap:
            return None

        # Determine gap type
        if signals['low_confidence'] and rag_response.retrieval_confidence < 0.2:
            gap_type = GapType.MISSING_KB.value
        elif signals['small_margin']:
            gap_type = GapType.INCOMPLETE_KB.value
        else:
            gap_type = GapType.OUTDATED_KB.value

        # Determine priority
        if signals['auto_zero'] or signals['repeated']:
            priority = "high"
        elif signal_count >= 2:
            priority = "medium"
        else:
            priority = "low"

        # Check if we should update existing KB
        should_update = False
        existing_kb_id = ""
        if rag_response.sources and rag_response.sources[0].document.doc_type == "KB":
            if gap_type == GapType.INCOMPLETE_KB.value:
                should_update = True
                existing_kb_id = rag_response.sources[0].document.source_id

        # Create gap record
        gap = KnowledgeGap(
            gap_type=gap_type,
            detected_from_query=rag_response.query[:500],
            retrieval_confidence=rag_response.retrieval_confidence,
            score_margin=rag_response.score_margin,
            repeated_count=self.issue_tracker[query_hash],
            triggered_by_auto_zero=signals['auto_zero'],
            topic=rag_response.query[:200],
            priority=priority,
            should_update_existing=should_update,
            existing_kb_to_update=existing_kb_id,
            status="open"
        )

        # Store in database
        self.db.insert("knowledge_gaps", gap.to_dict())

        # Log learning event
        self._log_event(
            event_type="gap_detected",
            gap_id=gap.gap_id,
            details={'signals': signals, 'priority': priority}
        )

        return gap

    def generate_kb_draft(
        self,
        ticket_number: str,
        gap: KnowledgeGap = None
    ) -> Optional[DraftKBArticle]:
        """
        Generate draft KB article from a resolved ticket.

        Args:
            ticket_number: Source ticket number
            gap: Optional associated knowledge gap

        Returns:
            DraftKBArticle if successful
        """
        # Get ticket data
        ticket = self.db.get("tickets", "Ticket_Number", ticket_number)
        if not ticket:
            return None

        # Get conversation
        conv = self.db.get("conversations", "Ticket_Number", ticket_number)
        transcript = conv.get('Transcript', '') if conv else ''

        # Generate content
        title = f"How to resolve: {ticket.get('Subject', 'Issue')}"

        content = f"""# {title}

## Problem Description
{ticket.get('Description', 'No description available.')}

## Root Cause
{ticket.get('Root_Cause', 'Root cause analysis pending.')}

## Solution

### Prerequisites
- Access to the {ticket.get('Product', 'system')} module
- Appropriate user permissions

### Steps to Resolve
{ticket.get('Resolution', 'Resolution steps not documented.')}

## Prevention
To prevent this issue in the future:
1. Regular system maintenance
2. Follow documented procedures
3. Contact support for guidance

## Related Information
- **Product:** {ticket.get('Product', 'N/A')}
- **Module:** {ticket.get('Module', 'N/A')}
- **Category:** {ticket.get('Category', 'N/A')}

## Tags
{ticket.get('Tags', 'support, troubleshooting')}

---
*Generated from Ticket: {ticket_number}*
*Generation Date: {datetime.now().isoformat()}*
"""

        # Determine if update or new
        is_update = gap.should_update_existing if gap else False
        updating_kb_id = gap.existing_kb_to_update if gap else ""

        # Create draft
        draft = DraftKBArticle(
            title=title,
            content=content,
            category=ticket.get('Category', ''),
            module=ticket.get('Module', ''),
            product=ticket.get('Product', ''),
            tags=ticket.get('Tags', '').split(',') if ticket.get('Tags') else [],
            source_tickets=[ticket_number],
            source_gap_id=gap.gap_id if gap else "",
            is_update=is_update,
            updating_kb_id=updating_kb_id,
            status="pending",
            generation_confidence=0.75
        )

        # Store in database
        self.db.insert("draft_kb_articles", {
            'draft_id': draft.draft_id,
            'title': draft.title,
            'content': draft.content,
            'category': draft.category,
            'module': draft.module,
            'product': draft.product,
            'tags': draft.tags,
            'source_tickets': draft.source_tickets,
            'source_gap_id': draft.source_gap_id,
            'is_update': draft.is_update,
            'updating_kb_id': draft.updating_kb_id,
            'status': draft.status,
            'generation_confidence': draft.generation_confidence,
            'created_at': draft.created_at
        })

        # Save to artifacts
        kb_versions_dir = Path(self.config.paths.kb_versions_dir)
        kb_versions_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = kb_versions_dir / f"{draft.draft_id}.md"
        artifact_path.write_text(content)

        # Log event
        self._log_event(
            event_type="draft_created",
            draft_id=draft.draft_id,
            ticket_number=ticket_number,
            gap_id=gap.gap_id if gap else "",
            details={'is_update': is_update}
        )

        # Update gap status
        if gap:
            self.db.execute(
                "UPDATE knowledge_gaps SET status = 'in_progress' WHERE gap_id = ?",
                [gap.gap_id]
            )

        return draft

    def submit_for_review(self, draft_id: str, reviewer: str = "qa_team") -> bool:
        """Submit draft for human review."""
        self.db.execute(
            """UPDATE draft_kb_articles
               SET status = 'in_review', reviewer = ?
               WHERE draft_id = ?""",
            [reviewer, draft_id]
        )
        self._log_event(
            event_type="review_started",
            draft_id=draft_id,
            details={'reviewer': reviewer}
        )
        return True

    def approve_draft(
        self,
        draft_id: str,
        reviewer: str,
        notes: str = ""
    ) -> Optional[str]:
        """
        Approve and publish a draft KB article.

        Args:
            draft_id: Draft to approve
            reviewer: Reviewer identifier
            notes: Review notes

        Returns:
            New KB article ID if successful
        """
        draft_row = self.db.get("draft_kb_articles", "draft_id", draft_id)
        if not draft_row:
            return None

        # Generate new KB article ID
        new_kb_id = f"KB_GEN_{draft_id[:8]}"

        # Create document
        doc = Document(
            doc_id=f"KB_{new_kb_id}",
            doc_type="KB",
            source_id=new_kb_id,
            title=draft_row['title'],
            content=draft_row['content'],
            metadata={
                'category': draft_row.get('category', ''),
                'module': draft_row.get('module', ''),
                'product': draft_row.get('product', ''),
                'tags': draft_row.get('tags', []),
                'source_type': 'auto_generated',
                'source_tickets': draft_row.get('source_tickets', [])
            },
            version=1
        )

        # Add to documents table
        self.db.insert("documents", {
            'doc_id': doc.doc_id,
            'doc_type': doc.doc_type,
            'source_id': doc.source_id,
            'title': doc.title,
            'content': doc.content,
            'metadata': doc.metadata,
            'version': 1,
            'created_at': datetime.now().isoformat(),
            'content_hash': doc.content_hash(),
            'is_active': 1
        })

        # Add to vector store
        self.vs.add_documents([doc])

        # Update draft status
        self.db.execute(
            """UPDATE draft_kb_articles
               SET status = 'approved',
                   review_notes = ?,
                   published_at = ?,
                   published_kb_id = ?
               WHERE draft_id = ?""",
            [notes, datetime.now().isoformat(), new_kb_id, draft_id]
        )

        # Create lineage record
        source_tickets = draft_row.get('source_tickets', [])
        if isinstance(source_tickets, str):
            try:
                source_tickets = json.loads(source_tickets) if source_tickets else []
            except json.JSONDecodeError:
                source_tickets = [source_tickets] if source_tickets else []

        for ticket_num in source_tickets:
            self.db.insert("kb_lineage", {
                'KB_Article_ID': new_kb_id,
                'Source_Type': 'Ticket',
                'Source_ID': ticket_num,
                'Relationship': 'Generated_From',
                'Evidence_Snippet': f"Auto-generated from resolved ticket {ticket_num}",
                'Event_Timestamp': datetime.now().isoformat()
            })

        # Update gap status
        if draft_row.get('source_gap_id'):
            self.db.execute(
                "UPDATE knowledge_gaps SET status = 'resolved' WHERE gap_id = ?",
                [draft_row['source_gap_id']]
            )

        # Log event
        self._log_event(
            event_type="published",
            draft_id=draft_id,
            kb_article_id=new_kb_id,
            details={'reviewer': reviewer, 'notes': notes}
        )

        return new_kb_id

    def reject_draft(self, draft_id: str, reviewer: str, reason: str) -> bool:
        """Reject a draft KB article."""
        self.db.execute(
            """UPDATE draft_kb_articles
               SET status = 'rejected', reviewer = ?, review_notes = ?
               WHERE draft_id = ?""",
            [reviewer, reason, draft_id]
        )
        self._log_event(
            event_type="rejected",
            draft_id=draft_id,
            details={'reviewer': reviewer, 'reason': reason}
        )
        return True

    def _log_event(
        self,
        event_type: str,
        gap_id: str = "",
        draft_id: str = "",
        kb_article_id: str = "",
        ticket_number: str = "",
        details: Dict = None
    ):
        """Log a learning event."""
        event_id = str(uuid.uuid4())

        self.db.insert("learning_events", {
            'Event_ID': event_id,
            'Trigger_Ticket_Number': ticket_number or '',
            'Trigger_Conversation_ID': '',
            'Detected_Gap': gap_id or '',
            'Proposed_KB_Article_ID': kb_article_id or '',
            'Draft_Summary': json.dumps(details) if details else '',
            'Final_Status': event_type,
            'Reviewer_Role': 'system',
            'Event_Timestamp': datetime.now().isoformat()
        })

        return event_id

    def get_pending_reviews(self) -> List[Dict]:
        """Get all drafts pending review."""
        return self.db.get_all(
            "draft_kb_articles",
            "status IN ('pending', 'in_review')"
        )

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning pipeline statistics."""
        stats = {}

        # Gap stats
        gaps = self.db.query("""
            SELECT status, COUNT(*) as count
            FROM knowledge_gaps
            GROUP BY status
        """)
        stats['gaps'] = {r['status']: r['count'] for r in gaps}

        # Draft stats
        drafts = self.db.query("""
            SELECT status, COUNT(*) as count
            FROM draft_kb_articles
            GROUP BY status
        """)
        stats['drafts'] = {r['status']: r['count'] for r in drafts}

        # Recent events
        events = self.db.query("""
            SELECT Final_Status as event_type, COUNT(*) as count
            FROM learning_events
            WHERE Event_Timestamp >= date('now', '-7 days')
            GROUP BY Final_Status
        """)
        stats['recent_events'] = {r['event_type']: r['count'] for r in events}

        return stats
