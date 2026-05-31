"""Pydantic models, types, events, and dataclasses for the legal assistant."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field
from config.app_settings import chat_settings as settings

from legal_pipeline.search_models import (  # noqa: F401 — re-exported for consumers
    ChunkModel,
    DocumentType,
    HydratedChunk,
    LegalDocumentModel,
)

ModeName = Literal["decide", "clarify", "retrieve", "user_doc_retrieve", "process_documents", "answer", "generate"]
QualityModeName = Literal["fast", "thorough"]


# --- Metadata Pydantic Models ---

class ModeRuns(BaseModel):
    """Counts of how many times each mode has been run."""
    model_config = ConfigDict(extra="forbid")
    decide: int = 0
    answer: int = 0
    retrieve: int = 0
    user_doc_retrieve: int = 0
    process_documents: int = 0
    clarify: int = 0
    generate: int = 0


class ToolCalls(BaseModel):
    """Counts of tool invocations."""
    model_config = ConfigDict(extra="forbid")
    set_mode: int = 0
    search_documents: int = 0


class RetrievalState(BaseModel):
    """State of the retrieval results."""
    model_config = ConfigDict(extra="forbid")
    results: List[Dict[str, Any]] = Field(default_factory=list)


class UserDocChunk(BaseModel):
    """A retrieved chunk from a user document."""
    model_config = ConfigDict(extra="forbid")
    chunk_index: int
    text: str
    score: float
    
    def for_llm(self) -> Dict[str, Any]:
        """Return LLM-friendly representation (exclude score)."""
        return {"chunk_index": self.chunk_index, "text": self.text}


class UserDoc(BaseModel):
    """Complete state of a single user document."""
    model_config = ConfigDict(extra="forbid")
    
    # Identity (from Django model)
    id: str
    filename: str
    
    # Processing state
    status: Literal["pending", "processing", "ready", "failed"]
    token_count: int = 0
    chunk_count: int = 0
    
    # Retrieval state for this document
    retrieved: bool = False
    weaviate_ingested: bool = False
    chunks: List[UserDocChunk] = Field(default_factory=list)
    summary: str = ""  # LLM summary of this doc's content
    
    def for_llm(self) -> Dict[str, Any]:
        """Return LLM-friendly representation (only filename, summary, chunks)."""
        return {
            "filename": self.filename,
            "summary": self.summary,
            "chunks": [c.for_llm() for c in self.chunks],
        }


class UserDocsState(BaseModel):
    """All user documents for the conversation."""
    model_config = ConfigDict(extra="forbid")
    documents: List[UserDoc] = Field(default_factory=list)


class UsageCall(BaseModel):
    """Token usage from a single LLM call."""
    model_config = ConfigDict(extra="forbid")
    model: str
    input_tokens: int
    output_tokens: int


class GeneratedDoc(BaseModel):
    """Reference to a generated document."""
    model_config = ConfigDict(extra="forbid")
    id: str
    filename: str
    format: str
    title: str
    announced: bool = False


class ChatMetadata(BaseModel):
    """Typed metadata for chat history - replaces untyped Dict[str, Any]."""
    model_config = ConfigDict(extra="forbid")
    
    # Mode tracking
    mode_steps: int = 0
    mode_runs: ModeRuns = Field(default_factory=ModeRuns)
    tool_calls: ToolCalls = Field(default_factory=ToolCalls)
    
    # Retrieval state
    retrieval_calls: int = 0
    retrieval_rounds: int = 0
    retrieval: RetrievalState = Field(default_factory=RetrievalState)
    
    # User documents state (replaces has_user_documents, has_pending_documents, user_doc_retrieval)
    user_docs: UserDocsState = Field(default_factory=UserDocsState)
    
    # Quality mode
    quality_mode: QualityModeName = "fast"
    
    # Conversation context
    conversation_id: str | None = None
    
    # Generated documents
    generated_documents: List[GeneratedDoc] = Field(default_factory=list)
    
    # Source usage tracking: maps work_ref_id -> user-turn count when last cited
    source_last_used: Dict[str, int] = Field(default_factory=dict)

    # Rolling conversation summary
    conversation_summary: str | None = None
    summary_up_to_index: int = 0

    # Retrieval coverage note (set by refine model)
    retrieval_coverage: str | None = None

    # Conversation chain for user-facing modes (answer, clarify)
    conversation_chain_id: str | None = None
    chain_message_count: int = 0
    chain_context_hash: str | None = None
    chain_last_mode: str | None = None
    chain_reused_last: bool = False


# --- Pydantic Payloads ---

class ClarificationPayload(BaseModel):
    """Payload for clarifying questions - single message field for streaming."""
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, description="Full markdown message with intro, questions, and general offer.")
    used_source_ids: List[str] = Field(
        description="work_ref_id of each legal document actually referenced in the clarification",
    )


class RetrievalQueryPayload(BaseModel):
    """Output from retrieval query generation with dual queries for hybrid search."""
    model_config = ConfigDict(extra="forbid")
    semantic_query: str = Field(min_length=1, description="Natural language query for vector search")
    lexical_query: str = Field(min_length=1, description="Keyword query for BM25 search")
    query_type: Literal["conceptual", "targeted"] = Field(
        description="conceptual: broad topic search. targeted: specific law/forskrift lookup by name.",
    )


class RetrievalQuerySet(BaseModel):
    """Multiple diverse query pairs for multi-query retrieval."""
    model_config = ConfigDict(extra="forbid")
    queries: List[RetrievalQueryPayload] = Field(min_length=1, max_length=7)




class RetrievalRefinePayload(BaseModel):
    """Output from retrieval refinement decision."""
    model_config = ConfigDict(extra="forbid")
    new_queries: List[RetrievalQueryPayload] = Field(
        max_length=7,
        description="New queries targeting MISSING sources. Empty list if satisfied.",
    )
    drop_work_ref_ids: List[str]
    coverage_summary: str = Field(
        max_length=200,
        description="Én setning: hvilke juridiske temaer de beholdte dokumentene dekker.",
    )


class AnswerPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(min_length=1)
    used_source_ids: List[str] = Field(
        description="work_ref_id of each legal document actually cited in the answer",
    )


class SummaryPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(min_length=1)


class GenerateDocumentPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, description="Dokumentets tittel")
    markdown: str = Field(
        min_length=1,
        description="Komplett dokument i markdown. KUN dokumentinnhold — ingen chatmeldinger, oppfølgingsspørsmål, eller rådgivende merknader.",
    )


class SetModeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: ModeName


class UserDocQueryPayload(BaseModel):
    """Output from user document query generation."""
    model_config = ConfigDict(extra="forbid")
    semantic_query: str = Field(min_length=1, description="Naturlig frase for å finne relevante klausuler")
    lexical_query: str = Field(min_length=1, description="Kontraktstermer og nøkkelord")


class UserDocQuerySet(BaseModel):
    """Multiple diverse query pairs for multi-query user document retrieval."""
    model_config = ConfigDict(extra="forbid")
    queries: List[UserDocQueryPayload] = Field(min_length=1, max_length=3)




# --- Chat History ---

class Message(BaseModel):
    """Chat message - compatible with pydantic-ai's message_history."""
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str


class ChatHistoryRaw(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conversation_history: List[Message] = Field(default_factory=list)

    def new_message(self, role: Literal["system", "user", "assistant"], content: str) -> Message:
        message = Message(role=role, content=content)
        self.conversation_history.append(message)
        return message


@dataclass
class ChatHistory:
    ui_chat_history_raw: ChatHistoryRaw = field(default_factory=ChatHistoryRaw)
    llm_chat_history_raw: ChatHistoryRaw = field(default_factory=ChatHistoryRaw)
    metadata: ChatMetadata = field(default_factory=ChatMetadata)
    usage_calls: list[UsageCall] = field(default_factory=list)
    mode: ModeName = "answer"


# --- Tool Call and Usage Tracking ---

@dataclass(frozen=True)
class ToolCall:
    name: str
    call_id: str
    arguments: dict


@dataclass(frozen=True)
class TrackedUsage:
    """Token usage tracked from an LLM call."""
    model: str
    input_tokens: int
    output_tokens: int


# --- Mode Result ---

@dataclass
class ModeResult:
    """Result from running a mode."""
    next_mode: Optional[ModeName]
    terminal: bool = False  # True if this mode ends the conversation turn


# --- Streaming Events ---

@dataclass(frozen=True)
class StatusEvent:
    """Progress status update for the client."""
    event: str = "status"
    message: str = ""
    mode: str = ""


@dataclass(frozen=True)
class ChunkEvent:
    """Streaming text chunk from the final answer."""
    event: str = "chunk"
    text: str = ""


@dataclass(frozen=True)
class ErrorEvent:
    """Error event."""
    event: str = "error"
    detail: str = ""


@dataclass(frozen=True)
class HeartbeatEvent:
    """Keepalive event to prevent SSE timeout during long thinking phases."""
    event: str = "heartbeat"


StreamEvent = Union[StatusEvent, ChunkEvent, DoneEvent, ErrorEvent, HeartbeatEvent]

HEARTBEAT_INTERVAL_SECONDS = 10
