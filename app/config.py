"""Configuration helpers for the Beauty Advisor application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""

    OPENAI = "openai"
    AZURE = "azure"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    provider: EmbeddingProvider
    model: str
    openai_api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_api_version: Optional[str] = None

    @classmethod
    def load(cls, env_path: Optional[Path] = None) -> "EmbeddingConfig":
        """Load configuration from environment variables / .env file."""

        if env_path is not None:
            load_dotenv(env_path)
        else:
            # Load from default .env in project root if present
            default_env = Path.cwd() / ".env"
            if default_env.exists():
                load_dotenv(default_env)

        provider = os.getenv("EMBEDDING_PROVIDER", EmbeddingProvider.OPENAI.value).lower()
        try:
            provider_enum = EmbeddingProvider(provider)
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise ValueError(
                "EMBEDDING_PROVIDER must be either 'openai' or 'azure'"
            ) from exc

        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

        if provider_enum is EmbeddingProvider.OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY must be set for the OpenAI embedding provider."
                )
            return cls(provider=provider_enum, model=model, openai_api_key=api_key)

        # Azure configuration
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        if not api_key or not endpoint:
            raise RuntimeError(
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set for the Azure provider."
            )
        if not model:
            raise RuntimeError(
                "EMBEDDING_MODEL must be set to an Azure deployment name for the Azure provider."
            )
        return cls(
            provider=provider_enum,
            model=model,
            openai_api_key=api_key,
            azure_endpoint=endpoint,
            azure_api_version=api_version,
        )

    def dump_env(self, env_path: Path) -> None:
        """Persist the configuration into a .env file."""

        lines = [f"EMBEDDING_PROVIDER={self.provider.value}", f"EMBEDDING_MODEL={self.model}"]
        if self.provider is EmbeddingProvider.OPENAI:
            if not self.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY missing; cannot write configuration")
            lines.append(f"OPENAI_API_KEY={self.openai_api_key}")
        else:
            if not (self.openai_api_key and self.azure_endpoint and self.azure_api_version):
                raise RuntimeError("Azure configuration incomplete; cannot write configuration")
            lines.extend(
                [
                    f"AZURE_OPENAI_API_KEY={self.openai_api_key}",
                    f"AZURE_OPENAI_ENDPOINT={self.azure_endpoint}",
                    f"AZURE_OPENAI_API_VERSION={self.azure_api_version}",
                ]
            )

        env_path = env_path.expanduser()
        env_path.parent.mkdir(parents=True, exist_ok=True)
        with env_path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
