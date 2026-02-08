"""
RAG (Retrieval Augmented Generation) Pipeline.
"""

import time
import uuid
from typing import List, Optional, Tuple

from supportmind.config.settings import get_config
from supportmind.config.prompts import PROMPTS
from supportmind.models.schemas import RAGResponse, RetrievalResult
from supportmind.pipelines.retrieval import UnifiedRetriever


class RAGGenerator:
    """
    RAG response generation with full traceability.
    """

    def __init__(self, retriever: UnifiedRetriever = None):
        """
        Initialize RAG generator.

        Args:
            retriever: UnifiedRetriever instance
        """
        self.retriever = retriever or UnifiedRetriever()
        self.config = get_config()

    def _format_context(
        self,
        results: List[RetrievalResult],
        max_chars: int = 4000
    ) -> str:
        """Format retrieved documents as context string."""
        context_parts = []
        total_chars = 0

        for r in results:
            doc = r.document
            citation = r.get_citation()

            # Truncate content if needed
            content = doc.content[:1000] if len(doc.content) > 1000 else doc.content

            part = f"""
{citation} (Score: {r.score:.3f}, Type: {doc.doc_type})
Title: {doc.title}
Content: {content}
"""
            if total_chars + len(part) > max_chars:
                break

            context_parts.append(part)
            total_chars += len(part)

        return "\n---\n".join(context_parts) if context_parts else "No relevant documents found."

    def _mock_llm_generate(
        self,
        context: str,
        question: str,
        sources: List[str]
    ) -> Tuple[str, str]:
        """
        Mock LLM response for demo without API.

        Args:
            context: Formatted context string
            question: User question
            sources: List of source citations

        Returns:
            Tuple of (response, confidence)
        """
        # Determine confidence based on source types
        has_kb = any('KB:' in s for s in sources)
        has_script = any('SCRIPT:' in s for s in sources)
        has_ticket = any('TICKET:' in s for s in sources)

        if has_kb:
            confidence = "HIGH"
            source_type = "knowledge base articles"
        elif has_script:
            confidence = "MEDIUM"
            source_type = "troubleshooting scripts"
        elif has_ticket:
            confidence = "MEDIUM"
            source_type = "similar resolved tickets"
        else:
            confidence = "LOW"
            source_type = "available documentation"

        response = f"""Based on the {source_type}, here is the answer to your question:

**Question:** {question[:100]}...

**Answer:**
The retrieved documentation provides guidance for this type of inquiry. Based on the sources found:

1. **Primary Resolution:** Review the referenced documentation for step-by-step instructions specific to your situation.

2. **Key Considerations:**
   - Ensure all prerequisites are met before proceeding
   - Follow the documented procedures in order
   - Contact support if the standard resolution doesn't apply

3. **Related Information:** The system identified {len(sources)} relevant source(s) that may contain additional helpful details.

**Next Steps:**
- Review the cited sources below for detailed instructions
- If the issue persists, consider escalation based on the troubleshooting scripts

[Confidence: {confidence}]
[Sources: {', '.join(sources[:3])}]"""

        return response, confidence

    def generate(
        self,
        query: str,
        conversation_history: str = ""
    ) -> RAGResponse:
        """
        Generate RAG response with full traceability.

        Args:
            query: User question
            conversation_history: Previous conversation context

        Returns:
            RAGResponse with answer, sources, confidence, and trace_id
        """
        start_time = time.time()

        # 1. Retrieve relevant documents
        results = self.retriever.retrieve(query)

        # 2. Detect answer type from evidence
        detected_type, type_confidence = self.retriever.detect_answer_type(results)

        # 3. Get confidence metrics
        retrieval_confidence = self.retriever.get_confidence(results)
        score_margin = self.retriever.get_score_margin(results)

        # 4. Format context for LLM
        context = self._format_context(results)

        # 5. Get source citations
        sources = [r.get_citation() for r in results]

        # 6. Generate response (mock or real LLM)
        if self.config.use_mock_llm:
            answer, answer_confidence = self._mock_llm_generate(context, query, sources)
        else:
            # Real LLM call would go here
            # answer = self._call_llm(context, query, conversation_history)
            answer, answer_confidence = self._mock_llm_generate(context, query, sources)

        # 7. Build response with full traceability
        processing_time = (time.time() - start_time) * 1000

        response = RAGResponse(
            query=query,
            answer=answer,
            sources=results,
            detected_type=detected_type,
            type_confidence=type_confidence,
            retrieval_confidence=retrieval_confidence,
            score_margin=score_margin,
            answer_confidence=answer_confidence,
            trace_id=str(uuid.uuid4()),
            processing_time_ms=processing_time
        )

        return response

    def generate_with_ticket_context(
        self,
        query: str,
        ticket_number: str
    ) -> RAGResponse:
        """
        Generate response with ticket context for agent copilot.

        Args:
            query: User question
            ticket_number: Associated ticket number

        Returns:
            RAGResponse with context-aware answer
        """
        # Get ticket details
        ticket = self.retriever.db.get("tickets", "Ticket_Number", ticket_number)

        # Get conversation
        conv = self.retriever.db.get("conversations", "Ticket_Number", ticket_number)

        # Build context
        history = ""
        if ticket:
            history += f"Ticket: {ticket.get('Subject', '')}\n"
            history += f"Product: {ticket.get('Product', '')}\n"
            history += f"Category: {ticket.get('Category', '')}\n"
        if conv:
            history += f"\nConversation Summary: {conv.get('Issue_Summary', '')}\n"

        return self.generate(query, history)
