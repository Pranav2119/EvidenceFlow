"""
config.py: Application configuration validated using Pydantic Settings.
Loads environment variables from .env with strict type validation.
"""

from pydantic import Field
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
        default="gemini-2.5-flash",
        description="Default Gemini model family for agent nodes",
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
    DATABASE_URL: str = Field(
        default="postgresql+psycopg://evidenceflow:evidenceflow_dev_secret@localhost:5432/evidenceflow_db",
        description="SQLAlchemy async/sync connection string",
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
    QDRANT_COLLECTION: str = Field(
        default="evidenceflow_passages",
        description="Primary passage chunk collection name",
    )

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


# Central singleton instance used across all modules
settings = Settings()
