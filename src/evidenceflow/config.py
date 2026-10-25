"""
config.py: Application configuration validated using Pydantic Settings.
Loads environment variables from .env with strict type validation.

Review fixes applied (2026-10-24):
  [1.1] DATABASE_URL computed from individual parts via @computed_field
  [1.2] require_gemini_key() helper for fail-fast API key validation
  [1.3] Lazy singleton via get_settings() instead of module-level instantiation
"""

from functools import lru_cache
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    EvidenceFlow AI centralized configuration schema.
    Enforces type checking and provides sensible defaults for local development.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore any unmapped extra variables in .env without throwing errors
    )

    # --- Gemini LLM Settings ---
    GEMINI_API_KEY: str = Field(
        default="",
        description="Google AI Studio Gemini API Key",
    )
    GEMINI_MODEL: str = Field(
        default="gemini-3.5-flash-lite",
        description="Default Gemini model family for agent nodes",
    )
    GEMINI_EMBEDDING_MODEL: str = Field(
        default="text-embedding-004",
        description="Gemini embedding model name (D2-2: configurable to avoid silent breakage on deprecation)",
    )

    # --- PostgreSQL Settings (Checkpointer, Step Ledger & Audit Trail) ---
    POSTGRES_USER: str = Field(
        default="evidenceflow",
        description="PostgreSQL username",
    )
    POSTGRES_PASSWORD: str = Field(
        default="evidenceflow_dev_secret",
        description="PostgreSQL password",
    )
    POSTGRES_DB: str = Field(
        default="evidenceflow_db",
        description="PostgreSQL database name",
    )
    POSTGRES_HOST: str = Field(
        default="localhost",
        description="PostgreSQL host",
    )
    POSTGRES_PORT: int = Field(
        default=5432,
        description="PostgreSQL port",
    )

    # [Fix 1.1] DATABASE_URL is now computed from individual parts, never drifts
    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        """SQLAlchemy connection string computed from individual Postgres fields."""
        return (
            f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # --- Embedding Provider Settings ---
    EMBEDDING_PROVIDER: str = Field(
        default="fastembed",
        description="Embedding provider: 'fastembed' (local 384-dim, 0 API key) or 'gemini' (768-dim, requires GEMINI_API_KEY)",
    )

    # --- Qdrant Vector Store Settings (Dense + Sparse Hybrid) ---
    QDRANT_HOST: str = Field(
        default="localhost",
        description="Qdrant host",
    )
    QDRANT_PORT: int = Field(
        default=6333,
        description="Qdrant HTTP REST API port",
    )
    QDRANT_GRPC_PORT: int = Field(
        default=6334,
        description="Qdrant gRPC port for high-throughput vector queries",
    )

    @computed_field
    @property
    def QDRANT_COLLECTION(self) -> str:
        """Dynamic collection name per embedding provider to prevent vector dimension mismatches."""
        return f"evidenceflow_passages_{self.EMBEDDING_PROVIDER}"

    # --- Neo4j Graph Database Settings (Conversational Memory & Contradictions) ---
    NEO4J_URI: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j Bolt binary protocol URI",
    )
    NEO4J_USER: str = Field(
        default="neo4j",
        description="Neo4j username",
    )
    NEO4J_PASSWORD: str = Field(
        default="evidenceflow_graph_secret",
        description="Neo4j password",
    )

    # --- Application Runtime Settings ---
    APP_ENV: str = Field(
        default="development",
        description="Runtime environment: development, test, or production",
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging verbosity level",
    )
    RETRIEVAL_LIMIT: int = Field(
        default=5,
        description="Max passages returned per retrieval query (D3-14: config-driven, not hardcoded in node)",
    )
    RETRIEVAL_SCORE_THRESHOLD: float = Field(
        default=0.60,
        description="Minimum vector similarity score cutoff; lower scores trigger clean abstention",
    )

    # [Fix 1.2] Fail-fast helper for LLM-dependent code paths
    def require_gemini_key(self) -> str:
        """Validates and returns the Gemini API key, raising immediately if not set."""
        if not self.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is required but not set. "
                "Add it to your .env file: GEMINI_API_KEY=your_key_here"
            )
        return self.GEMINI_API_KEY


# [Fix 1.3] Lazy singleton — .env file I/O only happens on first call, not at import time
@lru_cache
def get_settings() -> Settings:
    """Returns a cached Settings instance. Call this instead of importing a global."""
    return Settings()


# Backward-compatibility alias
settings: Settings = get_settings()
