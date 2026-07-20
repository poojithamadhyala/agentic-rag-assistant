"""Pydantic schemas for structured data handling and validation across the pipeline."""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class DocumentChunk(BaseModel):
    """A chunk of a source document stored in the vector store."""
    chunk_id: str
    doc_id: str
    source_path: str
    text: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)

    @field_validator("text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("chunk text cannot be empty")
        return v


class RetrievalResult(BaseModel):
    """A single retrieval hit with its similarity score."""
    chunk: DocumentChunk
    score: float


class Citation(BaseModel):
    """Attribution for a claim in an answer."""
    doc_id: str
    chunk_id: str


class AgentAnswer(BaseModel):
    """Structured, validated output of the agent."""
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"
    tools_used: list[str] = Field(default_factory=list)


class AskRequest(BaseModel):
    """API request body for /ask."""
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=20)


class AskResponse(BaseModel):
    """API response body for /ask."""
    question: str
    result: AgentAnswer
    retrieved: list[RetrievalResult]


class EvalCase(BaseModel):
    """A single evaluation case: question + expected evidence."""
    question: str
    expected_doc_ids: list[str]
    reference_answer: Optional[str] = None
