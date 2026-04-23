

def augment_prompt_with_context(query: str, search_results: List[Dict]) -> str:
    """
    Build augmented prompt with retrieved context for LLM.

    """
    # Assemble context from search results
    context_parts = []
    for i, result in enumerate(search_results, 1):
        context_parts.append(f"Source {i}: {result['metadata']}\n{result['content']}")
    
    context = "\n\n".join(context_parts)
    
    # Build augmented prompt
    augmented_prompt = f"""
Based on the following notebook chunk code classification examples, classify all the code chunks on this notebook.

EXAMPLES:
{context}

GIVE NOTEBOOK: {query}

Please provide a clear, accurate answer based on the examples above.
If the information is not available in the examples, try to find a good classification of the code chunk, without creating new code.
"""
    
    return augmented_prompt