"""
Agent Copilot Module.
Real-time assistance for support agents.
"""

from typing import Any, Dict, List, Optional

from supportmind.pipelines.rag import RAGGenerator
from supportmind.pipelines.quality import QAScorer
from supportmind.pipelines.learning import LearningPipeline


class AgentCopilot:
    """
    Live copilot for support agents.
    Provides real-time suggestions with confidence and sources.
    """

    def __init__(
        self,
        rag_generator: RAGGenerator = None,
        qa_scorer: QAScorer = None,
        learning_pipeline: LearningPipeline = None
    ):
        """
        Initialize agent copilot.

        Args:
            rag_generator: RAG generator instance
            qa_scorer: QA scorer instance
            learning_pipeline: Learning pipeline instance
        """
        self.rag = rag_generator or RAGGenerator()
        self.qa = qa_scorer or QAScorer()
        self.learning = learning_pipeline or LearningPipeline()

    def get_suggestion(
        self,
        customer_message: str,
        ticket_number: str = None,
        conversation_history: str = ""
    ) -> Dict[str, Any]:
        """
        Get AI suggestion for agent response.

        Args:
            customer_message: Latest customer message
            ticket_number: Associated ticket number (optional)
            conversation_history: Previous conversation context

        Returns:
            Dictionary containing:
            - suggested_response: Text for agent to use/modify
            - confidence: HIGH/MEDIUM/LOW
            - sources: List of source documents with citations
            - warnings: Any compliance warnings
            - qa_preview: Preview of QA score
        """
        # Generate RAG response
        if ticket_number:
            rag_response = self.rag.generate_with_ticket_context(
                customer_message, ticket_number
            )
        else:
            rag_response = self.rag.generate(customer_message, conversation_history)

        # Preview QA score
        context_str = "\n".join([s.document.title for s in rag_response.sources])
        qa_preview = self.qa.evaluate(
            question=customer_message,
            response=rag_response.answer,
            context=context_str
        )

        # Format sources
        sources = []
        for s in rag_response.sources[:3]:
            sources.append({
                'citation': s.get_citation(),
                'title': s.document.title,
                'type': s.document.doc_type,
                'score': round(s.score, 3),
                'snippet': s.document.content[:200] + "..."
            })

        # Compile warnings
        warnings = []
        if rag_response.retrieval_confidence < 0.4:
            warnings.append("⚠️ Low confidence - consider manual verification")
        if qa_preview.violations:
            warnings.append(f"⚠️ Compliance: {', '.join(qa_preview.violations)}")
        if rag_response.score_margin < 0.1:
            warnings.append("⚠️ Multiple similar sources - verify correct context")

        return {
            'suggested_response': rag_response.answer,
            'confidence': rag_response.answer_confidence,
            'retrieval_confidence': round(rag_response.retrieval_confidence, 3),
            'detected_type': rag_response.detected_type,
            'sources': sources,
            'warnings': warnings,
            'qa_preview': {
                'overall_score': round(qa_preview.overall_score, 1),
                'action': qa_preview.action,
                'suggestions': qa_preview.suggestions[:2]
            },
            'trace_id': rag_response.trace_id,
            'processing_time_ms': round(rag_response.processing_time_ms, 1)
        }

    def get_quick_answers(
        self,
        question: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get quick answer snippets without full generation.

        Args:
            question: User question
            top_k: Number of answers to return

        Returns:
            List of quick answer dictionaries
        """
        results = self.rag.retriever.retrieve(question, top_k=top_k)

        answers = []
        for r in results:
            answers.append({
                'citation': r.get_citation(),
                'title': r.document.title,
                'type': r.document.doc_type,
                'score': round(r.score, 3),
                'content': r.document.content[:500],
                'metadata': r.document.metadata
            })

        return answers

    def report_feedback(
        self,
        trace_id: str,
        feedback: str,
        was_helpful: bool,
        agent_id: str = ""
    ) -> bool:
        """
        Record agent feedback on suggestion.

        Args:
            trace_id: Trace ID of the suggestion
            feedback: Textual feedback
            was_helpful: Whether suggestion was helpful
            agent_id: Agent identifier

        Returns:
            True if feedback was recorded
        """
        self.learning._log_event(
            event_type="feedback_received",
            details={
                'trace_id': trace_id,
                'feedback': feedback,
                'helpful': was_helpful,
                'agent_id': agent_id
            }
        )
        return True

    def get_similar_tickets(
        self,
        ticket_number: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find similar resolved tickets for reference.

        Args:
            ticket_number: Current ticket number
            top_k: Number of similar tickets to find

        Returns:
            List of similar ticket summaries
        """
        # Get current ticket
        ticket = self.rag.retriever.db.get("tickets", "Ticket_Number", ticket_number)
        if not ticket:
            return []

        # Search for similar tickets
        query = f"{ticket.get('Subject', '')} {ticket.get('Description', '')[:200]}"
        results = self.rag.retriever.retrieve(query, top_k=top_k + 1, doc_type="TICKET")

        # Filter out current ticket and format
        similar = []
        for r in results:
            if r.document.source_id != ticket_number:
                ticket_data = self.rag.retriever.db.get(
                    "tickets", "Ticket_Number", r.document.source_id
                )
                if ticket_data:
                    similar.append({
                        'ticket_number': r.document.source_id,
                        'subject': ticket_data.get('Subject', ''),
                        'resolution': ticket_data.get('Resolution', '')[:300],
                        'similarity_score': round(r.score, 3),
                        'product': ticket_data.get('Product', ''),
                        'category': ticket_data.get('Category', '')
                    })

        return similar[:top_k]
