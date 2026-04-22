def zero_cost_router(query):
    query_lower = query.lower()
    
    # Check for GST shortcut
    if "gst" in query_lower:
        return "mistral:7b"
        
    # Check for complexity keywords
    complexity_keywords = ["lookup", "calculate", "format", "invoice", "math"]
    matches = sum(1 for k in complexity_keywords if k in query_lower)
    
    if matches >= 3:
        return "mistral:7b"
        
    return "tinyllama"
