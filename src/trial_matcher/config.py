"""Application configuration via pydantic-settings."""

from __future__ import annotations

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="TRIAL_MATCHER_",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM backend: "groq" (cloud, free) or "ollama" (local)
    llm_backend: str = "groq"

    # Groq (free cloud API — https://console.groq.com)
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # Ollama (local fallback)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_timeout_secs: int = 120

    # App mode
    demo_mode: bool = False

    # Confidence
    abstention_confidence_threshold: float = 0.35
    high_confidence_threshold: float = 0.80

    # External APIs
    ctgov_base_url: str = "https://clinicaltrials.gov/api/v2"

    # Data paths
    n2c2_data_dir: str = "data/raw/n2c2_2018"
    results_dir: str = "results"
    precomputed_dir: str = "data/precomputed"
    ontologies_dir: str = "data/ontologies"

    # Optional MIMIC-IV
    mimic_db_url: str | None = None

    @property
    def llm_configured(self) -> bool:
        """True if a live LLM backend is available."""
        if self.llm_backend == "groq":
            return bool(self.groq_api_key)
        return True  # ollama always assumed available locally


def get_settings() -> Settings:
    """Return settings, preferring Streamlit secrets if available."""
    try:
        import streamlit as st  # type: ignore[import]

        secrets = dict(st.secrets)
        flat: dict[str, str] = {}
        for k, v in secrets.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    flat[kk] = str(vv)
            else:
                flat[k] = str(v)
        for k, v in flat.items():
            os.environ.setdefault(k, v)
    except Exception:
        pass
    return Settings()
