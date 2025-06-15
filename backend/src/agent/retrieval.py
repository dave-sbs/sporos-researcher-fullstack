"""Retriever tool and query-enhancement helpers."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.tools import tool

from .configuration import get_vector_store


@tool
def retriever(query: str, k: int = 20, filters: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:  # noqa: D401,E501
    """Return (doc, score) tuples from Supabase.

    `filters` is a simple metadata-equality dict passed straight to
    `similarity_search_with_relevance_scores`.
    """

    vector_store = get_vector_store()
    if filters:
        # Exact-match metadata filtering
        similar = vector_store.similarity_search_with_relevance_scores(query, k, filter=filters)
    else:
        similar = vector_store.similarity_search_with_relevance_scores(query, k)
    return similar
