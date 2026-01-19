from __future__ import annotations

import os

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _running_on_cloud_run() -> bool:
    # Cloud Run services always set K_SERVICE; jobs may set CLOUD_RUN_JOB.
    return bool(os.getenv("K_SERVICE") or os.getenv("CLOUD_RUN_JOB"))


def _default_persistence_backend() -> str:
    # Cloud Run instances have an ephemeral filesystem, so SQLite-backed storage will
    # reset across revisions/instances. Default to Firestore when running on Cloud Run.
    return "firestore" if _running_on_cloud_run() else "sqlite"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_env: str = "dev"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    google_cloud_project: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_CLOUD_PROJECT", "GCP_PROJECT_ID"),
    )
    vertex_location: str = "us-central1"
    gemini_model: str = "gemini-2.5-flash"

    # Optional second-pass "judge" model to decide whether the draft answer is well-supported.
    # If enabled and confidence is below threshold, the service will trigger live retrieval/web search and retry.
    answer_judge_enabled: bool = True
    answer_judge_threshold: float = 0.55
    answer_judge_model: str | None = None

    rag_db_path: str = "./data/rag.sqlite"
    rag_top_k: int = 5

    # Google Programmable Search Engine (Custom Search JSON API) for live searches.
    # If configured, can be used as a fallback to find relevant wiki pages.
    google_cse_api_key: str | None = None
    google_cse_cx: str | None = None

    # Optional: scrape non-wiki pages returned by Google CSE.
    # This is "untrusted web" retrieval and may include ads/opinions; keep it off by default.
    web_scrape_enabled: bool = True
    web_scrape_max_pages: int = 3
    web_scrape_max_chunks_total: int = 6
    web_scrape_max_bytes: int = 800_000  # cap HTML bytes fetched per page
    web_scrape_timeout_s: float = 15.0

    # Optional: OSRSBox structured dataset ingestion (items/monsters).
    # When enabled, OSRSBox URLs are also allowlisted for citations.
    osrsbox_enabled: bool = True
    osrsbox_base_url: str = "https://www.osrsbox.com/osrsbox-db/"

    # Optional: OSRSBox static JSON DB as an additional structured source.
    # Enable only if you ingest OSRSBox docs into the local RAGStore.
    osrsbox_enabled: bool = True
    osrsbox_base_url: str = "https://www.osrsbox.com/osrsbox-db/"

    # YouTube Data API v3 (optional) for quest video search.
    youtube_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("YOUTUBE_API_KEY", "GCP_YT_API_KEY"),
    )
    youtube_max_results: int = 3
    youtube_max_summaries: int = 2

    # Path to RAG sources configuration JSON (repo root defaults to rag_sources.json).
    rag_sources_path: str | None = None

    # Fake "gold" donation counter. Defaults to RAG_DB_PATH if not set.
    gold_db_path: str | None = None

    # Gold persistence backend.
    # sqlite: local file (ephemeral on Cloud Run)
    # firestore: persistent in GCP Firestore
    gold_backend: str = Field(default_factory=_default_persistence_backend)  # sqlite|firestore

    # Public chat history log storage. Defaults to RAG_DB_PATH if not set.
    history_db_path: str | None = None

    # Public history persistence backend.
    # sqlite: local file (ephemeral on Cloud Run)
    # firestore: persistent in GCP Firestore
    history_backend: str = Field(default_factory=_default_persistence_backend)  # sqlite|firestore
    firestore_collection: str = "public_chat"

    # Firestore site-state storage (used for global gold total).
    firestore_site_collection: str = "site_state"
    firestore_gold_doc: str = "gold_total"

    paypal_env: str = "sandbox"  # sandbox|live
    paypal_client_id: str | None = None
    paypal_client_secret: str | None = None
    paypal_return_url: str = "http://localhost:8000/donate/success"
    paypal_cancel_url: str = "http://localhost:8000/donate/cancel"


settings = Settings()
