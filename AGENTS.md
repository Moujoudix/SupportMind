# AGENTS.md — SupportMind Guidelines (for AI coding agents)

This repository implements **SupportMind**, a self-learning support intelligence layer:
- Retrieval (KB/SCRIPT/TICKET/CONVERSATION)
- Grounded RAG responses with citations
- QA + compliance scoring
- Knowledge gap detection and KB drafting with lineage + versioning
- Analytics (Hit@K, MRR, trends)
- Streamlit dashboard

## Core principles
1) **Ground everything in sources**
   - Never invent steps or policies.
   - If the retrieved evidence is insufficient, ask clarifying questions or recommend escalation.
   - Every generated answer must include citations to the specific sources used.

2) **Traceability is mandatory**
   - Any answer/citation must reference the dataset identifiers:
     - KB: `KB_Article_ID`
     - Script: `Script_ID`
     - Ticket: `Ticket_Number`
     - Conversation: `Conversation_ID`
   - Any generated KB draft must include provenance links to source items (Ticket/Conversation/Script).
   - Prefer using `kb_lineage` rather than semantic guessing when available.

3) **Confidence must be consistent**
   - Only label confidence as `HIGH` if:
     - retrieval confidence is above the configured threshold, AND
     - score margin is above the configured threshold, AND
     - QA action is `ALLOW`, AND
     - no compliance flags are raised.
   - Otherwise use `MEDIUM` or `LOW` and include warnings/escalation recommendations.

4) **Respect doc types**
   - If the best evidence is a SCRIPT:
     - show required placeholders/inputs from `Placeholder_Dictionary`
     - do not fabricate real values for placeholders
   - If the best evidence is a KB article:
     - provide steps directly supported by the KB text
   - If the best evidence is a ticket/conversation:
     - treat as weaker authority; prefer drafting/updating KB rather than citing it as final policy.

## Engineering rules
- Keep diffs minimal and focused. Avoid refactors unless necessary.
- Add or update tests for bug fixes and metric changes.
- Prefer deterministic behavior and clear thresholds in `supportmind/config/settings.py`.
- Avoid introducing heavy dependencies unless absolutely needed.

## Required runnable commands (must remain working)
- `make test`
- `make demo`
- `make ingest` (should fail with a clear message if the workbook is missing)
- `make run-dashboard` (optional in CI)

## Evaluation expectations
When changing retrieval or metrics:
- Ensure `evaluate_retrieval_accuracy()` compares **normalized IDs** and respects `Questions.Answer_Type`.
- Provide a small, reproducible sample test/fixture that validates Hit@K and MRR.

When changing QA/compliance:
- QA must evaluate the answer against the **same retrieved chunks** used by generation (not titles only).
- Add explicit penalties for ungrounded or generic answers.

## Output formats (preferred)
- QA outputs should be structured (JSON-like dict) including:
  - scores (tone/accuracy/completeness/compliance/overall)
  - action (`allow` / `revise` / `escalate` / `block`)
  - flags (auto-zero reasons)
  - suggestions
  - trace_id
- RAG responses should include:
  - final answer text
  - citations list (IDs + titles)
  - retrieval confidence + margin
  - detected_type
  - trace_id

## Safety & privacy
- Do not add features that store secrets in the repo.
- Never log full raw customer data; store only synthetic or redacted samples in `tests/fixtures`.
