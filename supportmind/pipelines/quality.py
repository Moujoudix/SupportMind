"""
QA (Quality Assurance) and Compliance Scoring Pipeline.
"""

import re
from typing import List, Tuple

from supportmind.config.settings import get_config, QAAction
from supportmind.models.schemas import QAScore
from supportmind.stores.database import Database


class QAScorer:
    """
    QA evaluation with compliance checking.
    Produces structured scores and action recommendations.
    """

    def __init__(self, db: Database = None):
        """
        Initialize QA scorer.

        Args:
            db: Database instance for storing scores
        """
        self.db = db or Database()
        self.config = get_config()

        # PII patterns for compliance
        self.pii_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone
            r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',  # SSN
            r'\b\d{16}\b',  # Credit card
            r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14})\b',  # Visa/MC
        ]

        # Prohibited phrases
        self.prohibited_phrases = [
            "i promise", "guaranteed", "definitely will",
            "your fault", "stupid", "idiot",
            "confidential information", "password is"
        ]

    def _check_pii(self, text: str) -> List[str]:
        """Check for PII exposure."""
        violations = []

        for pattern in self.pii_patterns:
            if re.search(pattern, text):
                violations.append("Potential PII detected: pattern match")
                break

        return violations

    def _check_compliance(self, text: str) -> Tuple[List[str], bool]:
        """Check for compliance violations."""
        violations = []
        auto_zero = False
        text_lower = text.lower()

        # Check PII
        pii_violations = self._check_pii(text)
        if pii_violations:
            violations.extend(pii_violations)
            auto_zero = True

        # Check prohibited phrases
        for phrase in self.prohibited_phrases:
            if phrase in text_lower:
                violations.append(f"Prohibited phrase: '{phrase}'")
                if phrase in ["password is", "confidential information"]:
                    auto_zero = True

        return violations, auto_zero

    def _score_tone(self, response: str) -> float:
        """Score response tone (0-100)."""
        score = 70.0  # Base score

        # Positive indicators
        positive_words = [
            "thank", "please", "happy to help", "glad", "appreciate",
            "understand", "apologize", "sorry for"
        ]
        for word in positive_words:
            if word in response.lower():
                score += 5

        # Negative indicators
        negative_words = ["can't", "won't", "impossible", "never", "wrong"]
        for word in negative_words:
            if word in response.lower():
                score -= 5

        return max(0, min(100, score))

    def _score_accuracy(self, response: str, context: str) -> float:
        """Score factual accuracy based on context (0-100)."""
        if not context:
            return 50.0

        score = 60.0  # Base score

        # Check if response references context
        if any(cite in response for cite in ['[KB:', '[SCRIPT:', '[TICKET:']):
            score += 20

        # Check for hedging (good for accuracy when uncertain)
        hedging = ["based on", "according to", "the documentation shows"]
        for hedge in hedging:
            if hedge in response.lower():
                score += 5

        return max(0, min(100, score))

    def _score_completeness(self, question: str, response: str) -> float:
        """Score response completeness (0-100)."""
        score = 60.0

        # Length check
        if len(response) > 200:
            score += 10
        if len(response) > 500:
            score += 10

        # Structure check
        if "**" in response or "1." in response or "- " in response:
            score += 10

        # Next steps check
        if "next step" in response.lower() or "follow" in response.lower():
            score += 10

        return max(0, min(100, score))

    def evaluate(
        self,
        question: str,
        response: str,
        context: str = "",
        ticket_number: str = "",
        conversation_id: str = "",
        trace_id: str = ""
    ) -> QAScore:
        """
        Evaluate a response for QA scoring.

        Args:
            question: Original question
            response: Agent/AI response
            context: Retrieved context used
            ticket_number: Associated ticket
            conversation_id: Associated conversation
            trace_id: RAG trace ID

        Returns:
            QAScore with all metrics and action recommendation
        """
        # Check compliance first (can trigger auto-zero)
        violations, auto_zero = self._check_compliance(response)

        if auto_zero:
            # Auto-zero: all scores set to 0
            qa_score = QAScore(
                ticket_number=ticket_number,
                conversation_id=conversation_id,
                response_trace_id=trace_id,
                tone_score=0.0,
                accuracy_score=0.0,
                completeness_score=0.0,
                compliance_score=0.0,
                overall_score=0.0,
                auto_zero=True,
                auto_zero_reason="Compliance violation detected",
                violations=violations,
                action=QAAction.BLOCK.value,
                suggestions=[
                    "Review and remove sensitive information",
                    "Rewrite response following guidelines"
                ]
            )
        else:
            # Calculate individual scores
            tone = self._score_tone(response)
            accuracy = self._score_accuracy(response, context)
            completeness = self._score_completeness(question, response)
            compliance = 100.0 - (len(violations) * 20)
            compliance = max(0, compliance)

            # Weighted overall score
            overall = (
                0.2 * tone +
                0.5 * accuracy +
                0.2 * completeness +
                0.1 * compliance
            )

            # Determine action
            if overall >= 80:
                action = QAAction.ALLOW.value
            elif overall >= 60:
                action = QAAction.REVISE.value
            elif overall >= 40:
                action = QAAction.ESCALATE.value
            else:
                action = QAAction.BLOCK.value

            # Generate suggestions
            suggestions = []
            if tone < 70:
                suggestions.append("Improve tone: add empathy and positive language")
            if accuracy < 70:
                suggestions.append("Improve accuracy: cite sources more clearly")
            if completeness < 70:
                suggestions.append("Improve completeness: add next steps")
            if compliance < 100:
                suggestions.append(f"Address compliance issues: {'; '.join(violations)}")

            qa_score = QAScore(
                ticket_number=ticket_number,
                conversation_id=conversation_id,
                response_trace_id=trace_id,
                tone_score=tone,
                accuracy_score=accuracy,
                completeness_score=completeness,
                compliance_score=compliance,
                overall_score=overall,
                auto_zero=False,
                violations=violations,
                action=action,
                suggestions=suggestions
            )

        # Store in database
        self.db.insert("qa_scores", {
            'score_id': qa_score.score_id,
            'ticket_number': qa_score.ticket_number,
            'conversation_id': qa_score.conversation_id,
            'response_trace_id': qa_score.response_trace_id,
            'tone_score': qa_score.tone_score,
            'accuracy_score': qa_score.accuracy_score,
            'completeness_score': qa_score.completeness_score,
            'compliance_score': qa_score.compliance_score,
            'overall_score': qa_score.overall_score,
            'auto_zero': qa_score.auto_zero,
            'auto_zero_reason': qa_score.auto_zero_reason,
            'violations': qa_score.violations,
            'action': qa_score.action,
            'suggestions': qa_score.suggestions,
            'evaluated_at': qa_score.evaluated_at
        })

        return qa_score
