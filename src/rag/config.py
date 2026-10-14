"""Configuration centralisée, lue depuis l'environnement / .env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]

PROVIDERS = {
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "generation_model": "mistral-small-latest",
        "judge_model": "mistral-large-latest",
        "embedding_model": "mistral-embed",
    },
    "openai": {
        "base_url": None,
        "generation_model": "gpt-4o-mini",
        "judge_model": "gpt-4o",
        "embedding_model": "text-embedding-3-small",
    },
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    llm_provider: str = "mistral"
    mistral_api_key: str = ""
    openai_api_key: str = ""
    generation_model: str = ""
    judge_model: str = ""
    embedding_model: str = ""
    embedding_backend: str = "api"  # "api" | "local"

    corpus_dir: Path = ROOT / "corpus"
    chroma_dir: Path = ROOT / "data" / "chroma"
    collection_name: str = "altiscope"

    top_k: int = 5
    candidates_per_retriever: int = 20
    faithfulness_guardrail: bool = False
    faithfulness_threshold: float = 0.7
    rate_limit_per_10min: int = 20

    @property
    def provider(self) -> dict:
        if self.llm_provider not in PROVIDERS:
            raise ValueError(f"LLM_PROVIDER inconnu : {self.llm_provider}")
        return PROVIDERS[self.llm_provider]

    @property
    def api_key(self) -> str:
        key = self.mistral_api_key if self.llm_provider == "mistral" else self.openai_api_key
        if not key:
            raise RuntimeError(
                f"Clé API manquante pour {self.llm_provider} : renseigner "
                f"{self.llm_provider.upper()}_API_KEY dans .env"
            )
        return key

    @property
    def base_url(self) -> str | None:
        return self.provider["base_url"]

    @property
    def gen_model(self) -> str:
        return self.generation_model or self.provider["generation_model"]

    @property
    def eval_model(self) -> str:
        return self.judge_model or self.provider["judge_model"]

    @property
    def embed_model(self) -> str:
        return self.embedding_model or self.provider["embedding_model"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
