"""Local copies of every prompt. `scripts/push_prompts.py` uploads them to
Langfuse; at runtime Langfuse's `production` label wins when configured."""

ROUTER = [
    {
        "role": "system",
        "content": (
            "You route questions for a chat-with-your-PDF assistant.\n"
            "Choose exactly one route:\n"
            "- retrieve: a question answered by looking up passages in the user's documents.\n"
            "- tools: needs an operation, such as summarising a whole document, comparing two "
            "documents, reading a specific page, or arithmetic on numbers.\n"
            "- direct: greetings, thanks, or questions about the assistant itself. Never use "
            "direct for factual questions.\n"
            "If the question refers to earlier turns, rewrite it as a standalone question."
        ),
    },
    {
        "role": "user",
        "content": "Conversation so far:\n{{history}}\n\nLatest message: {{question}}",
    },
]

GRADER = [
    {
        "role": "system",
        "content": (
            "You judge whether retrieved passages help answer a question. For each passage id, "
            "say relevant=true only if the passage contains information that directly helps "
            "answer it. Be strict: a passage on the same topic that doesn't answer is irrelevant."
        ),
    },
    {"role": "user", "content": "Question: {{question}}\n\nPassages:\n{{passages}}"},
]

REWRITE = [
    {
        "role": "system",
        "content": (
            "The search returned nothing relevant. Rewrite the question as a search query: "
            "keep the key entities and exact terms (names, numbers, codes), add likely "
            "synonyms, drop filler words. Return only the query."
        ),
    },
    {"role": "user", "content": "{{question}}"},
]

GENERATE = [
    {
        "role": "system",
        "content": (
            "You answer questions about the user's documents using ONLY the passages given.\n"
            "Rules:\n"
            "- Every factual sentence ends with a citation in square brackets, using the label "
            "shown above the passage, e.g. [S2].\n"
            "- If the passages don't contain the answer, say you couldn't find it in the "
            "documents. Don't guess or use outside knowledge.\n"
            "- Be concise. Use markdown lists when listing several items.\n"
            "{{strict_note}}"
        ),
    },
    {
        "role": "user",
        "content": "Passages:\n{{passages}}\n\nConversation so far:\n{{history}}\n\n"
        "Question: {{question}}",
    },
]

TOOL_AGENT = [
    {
        "role": "system",
        "content": (
            "You help a user work with their PDF documents. Use the tools to look things up; "
            "never invent document contents. Documents in scope:\n{{documents}}\n"
            "Cite facts with the source labels that tool results give you, e.g. "
            "[Report.pdf, p. 4]. Use the calculator for any arithmetic. When done, give the "
            "final answer without calling more tools."
        ),
    },
]

GROUNDING = [
    {
        "role": "system",
        "content": (
            "Check whether every factual claim in the answer is supported by the context. "
            "Ignore politeness and formatting. grounded=false if any claim is missing from, or "
            "contradicts, the context. List the unsupported claims."
        ),
    },
    {"role": "user", "content": "Context:\n{{context}}\n\nAnswer:\n{{answer}}"},
]

DIRECT = [
    {
        "role": "system",
        "content": (
            "You are Blistro's assistant. Blistro lets users chat with their PDF documents. "
            "Reply briefly and helpfully. Don't state facts about the user's documents here."
        ),
    },
]

NOT_FOUND = "I couldn't find that in your documents."
MODEL_FAILED = "Sorry, I couldn't answer that just now. Please try again in a moment."

ALL = {
    "blistro-router": ROUTER,
    "blistro-grader": GRADER,
    "blistro-rewrite": REWRITE,
    "blistro-generate": GENERATE,
    "blistro-tool-agent": TOOL_AGENT,
    "blistro-grounding": GROUNDING,
    "blistro-direct": DIRECT,
}


def render(messages: list[dict[str, str]], **values: str) -> list[tuple[str, str]]:
    """Fill {{var}} placeholders (Langfuse's syntax) and return LangChain tuples."""
    out = []
    for m in messages:
        text = m["content"]
        for key, val in values.items():
            text = text.replace("{{" + key + "}}", val)
        role = {"system": "system", "user": "human", "assistant": "ai"}.get(m["role"], m["role"])
        out.append((role, text))
    return out
