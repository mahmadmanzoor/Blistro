from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ChunkStrategy = Literal["fixed", "recursive", "semantic"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: Literal["development", "test", "production"] = "development"

    # Postgres (psycopg 3 for everything; the LangGraph checkpointer requires it).
    # Plain libpq URL, e.g. postgresql://user:pass@host:5432/db
    database_url: str = "postgresql://blistro:blistro@localhost:55433/blistro_ai"
    # Set True behind PgBouncer / Neon pooler (transaction pooling): disables
    # server-side prepared statements, which those poolers can't keep per client.
    db_pooled: bool = False

    # Models
    openai_api_key: str | None = None
    chat_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dims: int = 1536
    # "openai" in real use; "fake" gives deterministic hash embeddings for tests.
    embeddings_provider: Literal["openai", "fake"] = "openai"

    # Retrieval
    active_strategy: ChunkStrategy = "recursive"
    ingest_strategies: list[ChunkStrategy] = Field(default_factory=lambda: ["recursive"])
    chunk_tokens: int = 512
    chunk_overlap: int = 64
    retrieval_k: int = 5
    retrieval_candidates: int = 30
    rrf_k: int = 60
    # Reranker: "cohere" (hosted, needs COHERE_API_KEY), "local" (cross-encoder on
    # this machine; `uv sync --extra local-rerank`), or "none".
    rerank_provider: Literal["cohere", "local", "none"] = "cohere"
    cohere_api_key: str | None = None
    rerank_model: str = "rerank-v5.0-fast"
    local_rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_enabled: bool = True

    # Agent guardrails
    agent_recursion_limit: int = 25
    llm_timeout_s: float = 45.0
    tool_timeout_s: float = 20.0

    # Auth: Kinde issues the tokens Blistro's Next.js app already holds.
    auth_mode: Literal["kinde", "dev"] = "kinde"
    kinde_issuer_url: str | None = None  # e.g. https://blistro.kinde.com
    kinde_audience: str | None = None
    # Shared secret for calls from Blistro's Next.js server, which has already
    # authenticated the user and passes their id in X-User-Id. Needed because
    # uploadthing's onUploadComplete runs without a user session to forward.
    service_token: str | None = None
    rate_limit_per_minute: int = 20

    # Langfuse (Cloud). Tracing and prompt management turn on when keys are set.
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # Ingestion
    max_pdf_bytes: int = 32 * 1024 * 1024
    # PDFs are fetched only from these hosts (uploadthing's CDNs by default), so a
    # caller can't make the worker fetch internal URLs (SSRF). Suffix match.
    allowed_source_hosts: list[str] = Field(
        default_factory=lambda: ["utfs.io", "ufs.sh", "uploadthing.com"]
    )
    worker_poll_s: float = 2.0
    job_max_attempts: int = 3

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    if s.env == "production" and s.auth_mode == "dev":
        raise RuntimeError("AUTH_MODE=dev is not allowed when ENV=production")
    if s.service_token is not None and len(s.service_token) < 32:
        raise RuntimeError("SERVICE_TOKEN must be at least 32 characters")
    return s
