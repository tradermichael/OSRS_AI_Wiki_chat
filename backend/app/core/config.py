from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_env: str = "dev"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    google_cloud_project: str | None = None
    vertex_location: str = "us-central1"
    gemini_model: str = "gemini-2.5-flash"

    rag_db_path: str = "./data/rag.sqlite"
    rag_top_k: int = 5

    # Path to RAG sources configuration JSON (repo root defaults to rag_sources.json).
    rag_sources_path: str | None = None

    # Fake "gold" donation counter. Defaults to RAG_DB_PATH if not set.
    gold_db_path: str | None = None

    # Public chat history log storage. Defaults to RAG_DB_PATH if not set.
    history_db_path: str | None = None

    paypal_env: str = "sandbox"  # sandbox|live
    paypal_client_id: str | None = None
    paypal_client_secret: str | None = None
    paypal_return_url: str = "http://localhost:8000/donate/success"
    paypal_cancel_url: str = "http://localhost:8000/donate/cancel"


settings = Settings()
