"""
Data Models / Schemas
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


@dataclass
class BaseModel:
    """Base class with serialization utilities."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for k, v in asdict(self).items():
            if isinstance(v, datetime):
                result[k] = v.isoformat()
            elif isinstance(v, Enum):
                result[k] = v.value
            elif v is not None:
                result[k] = v
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create instance from dictionary."""
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in field_names})


@dataclass
class Document(BaseModel):
    """Universal document for unified index."""
    doc_id: str
    doc_type: str  # KB, SCRIPT, TICKET, CONVERSATION
    source_id: str
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1
    created_at: str = ""
    updated_at: str = ""

    def get_searchable_text(self) -> str:
        """Get combined text for search/embedding."""
        parts = [self.title, self.content]
        for key in ['product', 'category', 'module', 'tags']:
            if key in self.metadata and self.metadata[key]:
                val = self.metadata[key]
                if isinstance(val, list):
                    parts.extend(val)
                else:
                    parts.append(str(val))
        return " ".join(filter(None, parts))

    def content_hash(self) -> str:
        """Compute content hash for change detection."""
        return hashlib.md5(f"{self.title}|{self.content}".encode()).hexdigest()


@dataclass
class RetrievalResult(BaseModel):
    """Search result with traceability."""
    document: Document
    score: float
    rank: int
    retrieval_method: str = "hybrid"
    semantic_score: float = 0.0
    keyword_score: float = 0.0

    def get_citation(self) -> str:
        """Generate citation string."""
        d = self.document
        return f"[{d.doc_type}:{d.source_id}]"


@dataclass
class RAGResponse(BaseModel):
    """Complete RAG response with traceability."""
    query: str
    answer: str
    sources: List[RetrievalResult] = field(default_factory=list)
    detected_type: str = "UNKNOWN"
    type_confidence: float = 0.0
    retrieval_confidence: float = 0.0
    score_margin: float = 0.0
    answer_confidence: str = "LOW"
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    processing_time_ms: float = 0.0

    def get_source_citations(self) -> List[str]:
        """Get all source citations."""
        return [s.get_citation() for s in self.sources]


@dataclass
class QAScore(BaseModel):
    """QA evaluation result."""
    score_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ticket_number: str = ""
    conversation_id: str = ""
    response_trace_id: str = ""
    tone_score: float = 0.0
    accuracy_score: float = 0.0
    completeness_score: float = 0.0
    compliance_score: float = 0.0
    overall_score: float = 0.0
    auto_zero: bool = False
    auto_zero_reason: str = ""
    violations: List[str] = field(default_factory=list)
    action: str = "allow"
    suggestions: List[str] = field(default_factory=list)
    evaluated_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class KnowledgeGap(BaseModel):
    """Detected knowledge gap."""
    gap_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    gap_type: str = "missing_kb"
    detected_from_ticket: str = ""
    detected_from_query: str = ""
    retrieval_confidence: float = 0.0
    score_margin: float = 0.0
    repeated_count: int = 1
    triggered_by_auto_zero: bool = False
    topic: str = ""
    priority: str = "medium"
    should_update_existing: bool = False
    existing_kb_to_update: str = ""
    status: str = "open"
    detected_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class DraftKBArticle(BaseModel):
    """Draft KB article awaiting review."""
    draft_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    content: str = ""
    category: str = ""
    module: str = ""
    product: str = ""
    tags: List[str] = field(default_factory=list)
    source_tickets: List[str] = field(default_factory=list)
    source_gap_id: str = ""
    is_update: bool = False
    updating_kb_id: str = ""
    status: str = "pending"
    reviewer: str = ""
    review_notes: str = ""
    generation_confidence: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    published_kb_id: str = ""


@dataclass
class LearningEvent(BaseModel):
    """Learning pipeline event - matches CSV schema."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trigger_ticket_number: str = ""
    trigger_conversation_id: str = ""
    detected_gap: str = ""
    proposed_kb_article_id: str = ""
    draft_summary: str = ""
    final_status: str = ""
    reviewer_role: str = ""
    event_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class KBLineage(BaseModel):
    """KB provenance - matches CSV schema."""
    kb_article_id: str = ""
    source_type: str = ""
    source_id: str = ""
    relationship: str = ""
    evidence_snippet: str = ""
    event_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
