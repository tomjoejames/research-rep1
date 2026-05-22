def zero_cost_router(query):
    query_lower = query.lower()

    # Check for GST shortcut
    if "gst" in query_lower:
        return "mistral:7b"

    # Check for complexity keywords

    # Routing threshold is >= 3 keyword matches.
    # "gst" is intentionally scored here rather than short-circuited above,
    # so a prompt needs GST *plus* at least 2 other complexity signals to escalate.
    # SINGLE_PROMPT scores 2 ("calculate" + "gst") → single-step.
    # AGENT_STEP prompts score 4+ ("lookup", "calculate", "format", "invoice") → multi-agent.

    complexity_keywords = ["lookup", "calculate", "format", "invoice", "math", "gst"]
    matches = sum(1 for k in complexity_keywords if k in query_lower)

    if matches >= 3:
        return "mistral:7b"

    return "tinyllama"
