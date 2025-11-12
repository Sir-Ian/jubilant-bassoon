"""Interactive setup utility for configuring embedding providers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from ..config import EmbeddingConfig, EmbeddingProvider


def prompt_input(prompt: str, *, default: Optional[str] = None, secret: bool = False) -> str:
    """Prompt for input with optional default."""

    suffix = f" [{default}]" if default else ""
    full_prompt = f"{prompt}{suffix}: "
    if secret:
        import getpass

        value = getpass.getpass(full_prompt)
    else:
        value = input(full_prompt)
    value = value.strip()
    if not value and default is not None:
        return default
    return value


def build_config(provider: EmbeddingProvider) -> EmbeddingConfig:
    """Interactively gather configuration for a provider."""

    if provider is EmbeddingProvider.OPENAI:
        api_key = prompt_input("OpenAI API key", secret=True)
        model = prompt_input(
            "OpenAI embedding model",
            default="text-embedding-3-large",
        )
        return EmbeddingConfig(
            provider=provider,
            model=model,
            openai_api_key=api_key,
        )

    # Azure provider configuration
    api_key = prompt_input("Azure OpenAI API key", secret=True)
    endpoint = prompt_input("Azure OpenAI endpoint (https://<resource>.openai.azure.com)")
    deployment = prompt_input("Azure embedding deployment name")
    api_version = prompt_input(
        "Azure OpenAI API version",
        default="2024-02-15-preview",
    )
    return EmbeddingConfig(
        provider=provider,
        model=deployment,
        openai_api_key=api_key,
        azure_endpoint=endpoint,
        azure_api_version=api_version,
    )


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=[p.value for p in EmbeddingProvider],
        help="Embedding provider to configure (default: openai)",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to write the resulting .env file",
    )
    args = parser.parse_args(argv)

    provider = EmbeddingProvider(args.provider or EmbeddingProvider.OPENAI.value)
    config = build_config(provider)
    config.dump_env(args.env_file)
    print(f"Configuration saved to {args.env_file}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
