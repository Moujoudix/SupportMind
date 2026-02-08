"""
LLM Prompt Templates
"""

PROMPTS = {
    "rag_system": """You are SupportMind, an expert AI support assistant.

RULES:
1. Base answers ONLY on provided context documents
2. If context is insufficient, say "I don't have enough information"
3. ALWAYS cite sources using [DOC_ID] format
4. Never fabricate information
5. Be professional, empathetic, and clear""",

    "rag_user": """RETRIEVED DOCUMENTS:
{context}

QUESTION: {question}

CONVERSATION HISTORY:
{history}

Provide a helpful response based ONLY on the retrieved documents.
End with: [Confidence: HIGH/MEDIUM/LOW] [Sources: list DOC_IDs used]""",

    "kb_generation": """Create a KB article from this resolved ticket:

TICKET: {ticket_number}
PRODUCT: {product}
MODULE: {module}
CATEGORY: {category}
SUBJECT: {subject}
DESCRIPTION: {description}
RESOLUTION: {resolution}
ROOT_CAUSE: {root_cause}

CONVERSATION TRANSCRIPT:
{transcript}

Generate a KB article with:
1. **Title**: Clear, searchable
2. **Problem**: What issue was faced
3. **Root Cause**: Why it happened
4. **Solution**: Step-by-step resolution
5. **Prevention**: How to avoid in future
6. **Tags**: Relevant keywords""",

    "qa_evaluation": """Evaluate this support response:

QUESTION: {question}
RESPONSE: {response}
CONTEXT: {context}

Return JSON:
{{
    "tone_score": <0-100>,
    "accuracy_score": <0-100>,
    "completeness_score": <0-100>,
    "compliance_score": <0-100>,
    "overall_score": <0-100>,
    "auto_zero": <true/false>,
    "auto_zero_reason": "<reason>",
    "violations": [],
    "suggestions": [],
    "action": "<allow/revise/escalate/block>"
}}""",

    "gap_detection": """Analyze for knowledge gaps:

QUERY: {query}
TOP_CONFIDENCE: {top_confidence}
SCORE_MARGIN: {score_margin}
RESOLUTION: {resolution}

Return JSON:
{{
    "has_gap": <true/false>,
    "gap_type": "<missing_kb/outdated_kb/incomplete_kb/none>",
    "topic": "<suggested topic>",
    "priority": "<high/medium/low>",
    "should_update_existing": <true/false>,
    "existing_kb_to_update": "<KB_ID or null>",
    "reasoning": "<explanation>"
}}"""
}
