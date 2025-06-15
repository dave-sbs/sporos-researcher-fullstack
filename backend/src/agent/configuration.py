from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Optional

from dotenv import load_dotenv
from supabase import Client, create_client  # type: ignore

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
if not SUPABASE_URL:
    raise ValueError("SUPABASE_URL not found in environment variables")
if not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("SUPABASE_SERVICE_ROLE_KEY not found in environment variables")

@lru_cache(maxsize=1)
def get_supabase_client() -> Client:  # pragma: no cover
    """Return a cached Supabase client instance."""
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)  # type: ignore[arg-type]


@lru_cache(maxsize=1)
def get_vector_store(table_name: str = "chunks_test2") -> SupabaseVectorStore:  # pragma: no cover
    """Return a cached `SupabaseVectorStore` bound to *table_name*."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return SupabaseVectorStore(
        client=get_supabase_client(),
        embedding=embeddings,
        table_name=table_name,
        query_name="search_bill_chunks_langchain",
    )


@lru_cache(maxsize=4)
def get_llm(model: str = "gpt-4o-mini"):
    """Return (and cache) an LLM created via `init_chat_model`. Keyed by model name."""
    return init_chat_model(model=model)


class Configuration(BaseModel):
    """The configuration for the agent."""

    query_generator_model: str = Field(
        default="gpt-4o-mini",
        metadata={
            "description": "The name of the language model to use for the agent's query generation and document grading."
        },
    )

    answer_model: str = Field(
        default="gpt-4o",
        metadata={
            "description": "The name of the language model to use for generating answers."
        },
    )


    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)
