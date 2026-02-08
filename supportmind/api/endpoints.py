"""
FastAPI REST API endpoints.
"""

from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from supportmind.pipelines.rag import RAGGenerator
from supportmind.pipelines.quality import QAScorer
from supportmind.pipelines.learning import LearningPipeline
from supportmind.api.copilot import AgentCopilot
from supportmind.analytics.metrics import Analytics


# Initialize FastAPI app
app = FastAPI(
    title="SupportMind API",
    description="Self-Learning AI Support Intelligence System",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components (lazy loading)
_components = {}


def get_components():
    """Lazy load components."""
    if not _components:
        _components['rag'] = RAGGenerator()
        _components['qa'] = QAScorer()
        _components['learning'] = LearningPipeline()
        _components['copilot'] = AgentCopilot(
            _components['rag'],
            _components['qa'],
            _components['learning']
        )
        _components['analytics'] = Analytics()
    return _components


# Request/Response Models
class QueryRequest(BaseModel):
    question: str
    conversation_history: Optional[str] = ""
    ticket_number: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    confidence: str
    sources: list
    trace_id: str
    detected_type: str
    processing_time_ms: float


class FeedbackRequest(BaseModel):
    trace_id: str
    feedback: str
    was_helpful: bool
    agent_id: Optional[str] = ""


class QARequest(BaseModel):
    question: str
    response: str
    context: Optional[str] = ""


# Endpoints
@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "SupportMind API"}


@app.get("/health")
async def health():
    """Detailed health check."""
    components = get_components()
    health_data = components['analytics'].get_system_health()
    return health_data


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the knowledge base and get an AI-generated response.
    """
    components = get_components()

    if request.ticket_number:
        response = components['rag'].generate_with_ticket_context(
            request.question,
            request.ticket_number
        )
    else:
        response = components['rag'].generate(
            request.question,
            request.conversation_history
        )

    return QueryResponse(
        answer=response.answer,
        confidence=response.answer_confidence,
        sources=[s.get_citation() for s in response.sources],
        trace_id=response.trace_id,
        detected_type=response.detected_type,
        processing_time_ms=response.processing_time_ms
    )


@app.post("/copilot/suggest")
async def copilot_suggest(request: QueryRequest):
    """
    Get agent copilot suggestion.
    """
    components = get_components()

    suggestion = components['copilot'].get_suggestion(
        customer_message=request.question,
        ticket_number=request.ticket_number,
        conversation_history=request.conversation_history or ""
    )

    return suggestion


@app.post("/copilot/feedback")
async def copilot_feedback(request: FeedbackRequest):
    """
    Submit feedback on a copilot suggestion.
    """
    components = get_components()

    success = components['copilot'].report_feedback(
        trace_id=request.trace_id,
        feedback=request.feedback,
        was_helpful=request.was_helpful,
        agent_id=request.agent_id or ""
    )

    return {"success": success}


@app.post("/qa/evaluate")
async def qa_evaluate(request: QARequest):
    """
    Evaluate a response for QA scoring.
    """
    components = get_components()

    score = components['qa'].evaluate(
        question=request.question,
        response=request.response,
        context=request.context or ""
    )

    return score.to_dict()


@app.get("/analytics/dashboard")
async def analytics_dashboard():
    """
    Get dashboard data.
    """
    components = get_components()
    return components['analytics'].get_dashboard_data()


@app.get("/analytics/retrieval-accuracy")
async def retrieval_accuracy(sample_size: int = Query(default=50, ge=1, le=500)):
    """
    Evaluate retrieval accuracy.
    """
    components = get_components()
    return components['analytics'].evaluate_retrieval_accuracy(sample_size)


@app.get("/learning/pending-reviews")
async def pending_reviews():
    """
    Get pending KB draft reviews.
    """
    components = get_components()
    return components['learning'].get_pending_reviews()


@app.get("/learning/stats")
async def learning_stats():
    """
    Get learning pipeline statistics.
    """
    components = get_components()
    return components['learning'].get_learning_stats()


@app.post("/learning/approve/{draft_id}")
async def approve_draft(draft_id: str, reviewer: str = "api_user", notes: str = ""):
    """
    Approve a draft KB article.
    """
    components = get_components()

    kb_id = components['learning'].approve_draft(draft_id, reviewer, notes)

    if not kb_id:
        raise HTTPException(status_code=404, detail="Draft not found")

    return {"success": True, "kb_article_id": kb_id}


@app.post("/learning/reject/{draft_id}")
async def reject_draft(draft_id: str, reviewer: str = "api_user", reason: str = ""):
    """
    Reject a draft KB article.
    """
    components = get_components()

    success = components['learning'].reject_draft(draft_id, reviewer, reason)

    return {"success": success}
