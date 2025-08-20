"""Dynamic model configuration loader.

This module loads model configurations from external JSON files,
providing a dynamic alternative to hard-coded model lists.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    name: str
    provider: str
    family: str
    tier: str
    status: str = "ga"
    context_window: int = 4096
    max_output_tokens: int = 4096
    supports_tools: bool = False
    supports_vision: bool = False
    supports_streaming: bool = True
    supports_thinking: bool = False
    supports_reasoning: bool = False
    supports_grammar: bool = False
    supports_reasoning_effort: bool = False
    requires_max_completion_tokens: bool = False
    supports_max_completion_tokens: bool = False
    supports_computer_use: bool = False
    supports_realtime: bool = False
    supports_audio: bool = False
    supports_audio_input: bool = False
    pricing_tier: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class ProviderConfig:
    """Configuration for a provider."""

    name: str
    endpoint: str
    models: dict[str, ModelConfig]
    defaults: dict[str, str]


class ModelConfigLoader:
    """Loads and manages model configurations from external files."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the config loader.

        Args:
            config_path: Path to the model configuration JSON file.
                        If None, uses the default config file.
        """
        self.config_path = config_path or self._get_default_config_path()
        self._config: Optional[dict[str, Any]] = None
        self._providers: Optional[dict[str, ProviderConfig]] = None

    def _get_default_config_path(self) -> Path:
        """Get the default configuration file path."""
        return Path(__file__).parent / "models.json"

    def load(self, force_reload: bool = False) -> dict[str, Any]:
        """Load the configuration from the JSON file.

        Args:
            force_reload: If True, reload even if already loaded.

        Returns:
            The loaded configuration dictionary.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            json.JSONDecodeError: If the config file is invalid JSON.
        """
        if self._config is not None and not force_reload:
            return self._config

        try:
            with self.config_path.open(encoding="utf-8") as f:
                self._config = json.load(f)

            logger.info(f"Loaded model configuration from {self.config_path}")
            self._providers = None  # Clear cached providers
            return self._config

        except FileNotFoundError:
            logger.exception(f"Model configuration file not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.exception(f"Invalid JSON in model configuration: {e}")
            raise

    def get_providers(self) -> dict[str, ProviderConfig]:
        """Get all provider configurations.

        Returns:
            Dictionary mapping provider names to ProviderConfig objects.
        """
        if self._providers is not None:
            return self._providers

        config = self.load()
        self._providers = {}

        for provider_name, provider_data in config["providers"].items():
            models = {}

            for model_name, model_data in provider_data["models"].items():
                model_config = ModelConfig(
                    name=model_name,
                    provider=provider_name,
                    family=model_data.get("family", "unknown"),
                    tier=model_data.get("tier", "standard"),
                    status=model_data.get("status", "ga"),
                    context_window=model_data.get("context_window", 4096),
                    max_output_tokens=model_data.get("max_output_tokens", 4096),
                    supports_tools=model_data.get("supports_tools", False),
                    supports_vision=model_data.get("supports_vision", False),
                    supports_streaming=model_data.get("supports_streaming", True),
                    supports_thinking=model_data.get("supports_thinking", False),
                    supports_reasoning=model_data.get("supports_reasoning", False),
                    supports_grammar=model_data.get("supports_grammar", False),
                    supports_reasoning_effort=model_data.get("supports_reasoning_effort", False),
                    requires_max_completion_tokens=model_data.get("requires_max_completion_tokens", False),
                    supports_max_completion_tokens=model_data.get("supports_max_completion_tokens", False),
                    supports_computer_use=model_data.get("supports_computer_use", False),
                    supports_realtime=model_data.get("supports_realtime", False),
                    supports_audio=model_data.get("supports_audio", False),
                    supports_audio_input=model_data.get("supports_audio_input", False),
                    pricing_tier=model_data.get("pricing_tier"),
                    notes=model_data.get("notes"),
                )
                models[model_name] = model_config

            provider_config = ProviderConfig(
                name=provider_data["name"],
                endpoint=provider_data["endpoint"],
                models=models,
                defaults=provider_data.get("defaults", {}),
            )
            self._providers[provider_name] = provider_config

        return self._providers

    def get_models_by_provider(self, provider: str) -> list[str]:
        """Get all model names for a specific provider.

        Args:
            provider: The provider name (e.g., 'openai', 'anthropic').

        Returns:
            List of model names for the provider.
        """
        providers = self.get_providers()
        if provider not in providers:
            return []

        return list(providers[provider].models.keys())

    def get_models_by_family(self, family: str) -> list[str]:
        """Get all model names in a specific family.

        Args:
            family: The model family (e.g., 'gpt-5', 'claude-4').

        Returns:
            List of model names in the family.
        """
        providers = self.get_providers()
        models = []

        for provider_config in providers.values():
            for model_name, model_config in provider_config.models.items():
                if model_config.family == family:
                    models.append(model_name)

        return models

    def get_models_by_capability(self, capability: str) -> list[str]:
        """Get all models that support a specific capability.

        Args:
            capability: The capability name (e.g., 'tools', 'vision', 'thinking').

        Returns:
            List of model names that support the capability.
        """
        providers = self.get_providers()
        models = []

        capability_attr = f"supports_{capability}"

        for provider_config in providers.values():
            for model_name, model_config in provider_config.models.items():
                if hasattr(model_config, capability_attr) and getattr(model_config, capability_attr):
                    models.append(model_name)

        return models

    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model.

        Args:
            model_name: The model name.

        Returns:
            ModelConfig object or None if not found.
        """
        providers = self.get_providers()

        for provider_config in providers.values():
            if model_name in provider_config.models:
                return provider_config.models[model_name]

        return None

    def get_default_models(self) -> dict[str, str]:
        """Get default models for each provider.

        Returns:
            Dictionary mapping provider names to default model names.
        """
        providers = self.get_providers()
        defaults = {}

        for provider_name, provider_config in providers.items():
            # Use chat model as the overall default
            if "chat" in provider_config.defaults:
                defaults[provider_name] = provider_config.defaults["chat"]
            elif provider_config.models:
                # Fallback to first model if no default specified
                defaults[provider_name] = next(iter(provider_config.models.keys()))

        return defaults

    def get_anthropic_model_groups(self) -> dict[str, list[str]]:
        """Generate Anthropic model categorization from loaded configuration."""
        anthropic_models = self.get_models_by_provider("anthropic")

        groups = {
            "claude_4": [],
            "claude_3_7": [],
            "claude_3_5": [],
            "claude_3": [],
            "sonnet": [],
            "opus": [],
            "haiku": [],
        }

        for model_name in anthropic_models:
            model_config = self.get_model_config(model_name)
            if not model_config:
                continue

            # Categorize by family and tier
            family = model_config.family.lower()
            model_name_lower = model_name.lower()

            # Categorize by family
            if "claude-4" in family:
                groups["claude_4"].append(model_name)
            elif "claude-3.7" in family:
                groups["claude_3_7"].append(model_name)
            elif "claude-3.5" in family:
                groups["claude_3_5"].append(model_name)
            elif "claude-3" in family:
                groups["claude_3"].append(model_name)

            # Also categorize by model tier (check model name, not just family)
            if "sonnet" in model_name_lower:
                groups["sonnet"].append(model_name)
            elif "opus" in model_name_lower:
                groups["opus"].append(model_name)
            elif "haiku" in model_name_lower:
                groups["haiku"].append(model_name)

        return groups

    def get_openai_model_groups(self) -> dict[str, list[str]]:
        """Generate OpenAI model categorization from loaded configuration."""
        openai_models = self.get_models_by_provider("openai")

        groups = {
            "gpt_5": [],
            "gpt_4o": [],
            "gpt_4": [],
            "gpt_4_1": [],
            "gpt_3_5": [],
            "o_series": [],
            "audio": [],
            "computer_use": [],
            "realtime": [],
        }

        for model_name in openai_models:
            model_config = self.get_model_config(model_name)
            if not model_config:
                continue

            # Categorize by family (order matters - more specific first!)
            family = model_config.family.lower()
            if family.startswith("gpt-5"):
                groups["gpt_5"].append(model_name)
            elif family.startswith("gpt-4.1"):  # Must come before gpt-4!
                groups["gpt_4_1"].append(model_name)
            elif family.startswith("gpt-4o"):
                groups["gpt_4o"].append(model_name)
            elif family.startswith("gpt-4"):
                groups["gpt_4"].append(model_name)
            elif family.startswith("gpt-3.5"):
                groups["gpt_3_5"].append(model_name)
            elif family.startswith(("o1", "o3", "o4")):
                groups["o_series"].append(model_name)

            # Special capabilities
            if "audio" in family:
                groups["audio"].append(model_name)
            if "computer-use" in family:
                groups["computer_use"].append(model_name)
            if "realtime" in family:
                groups["realtime"].append(model_name)

        return groups


# Global loader instance
_global_loader: Optional[ModelConfigLoader] = None


def get_global_loader() -> ModelConfigLoader:
    """Get the global model configuration loader."""
    global _global_loader
    if _global_loader is None:
        _global_loader = ModelConfigLoader()
    return _global_loader


def load_model_config(config_path: Optional[Path] = None, force_reload: bool = False) -> dict[str, Any]:
    """Load model configuration from external file.

    Args:
        config_path: Path to configuration file. Uses default if None.
        force_reload: Whether to force reload even if already loaded.

    Returns:
        The loaded configuration dictionary.
    """
    loader = ModelConfigLoader(config_path) if config_path is not None else get_global_loader()

    return loader.load(force_reload=force_reload)


def get_models_by_provider(provider: str) -> list[str]:
    """Get all model names for a provider using the global loader.

    Args:
        provider: The provider name.

    Returns:
        List of model names.
    """
    return get_global_loader().get_models_by_provider(provider)


def get_model_families() -> set[str]:
    """Get all available model families.

    Returns:
        Set of family names.
    """
    loader = get_global_loader()
    providers = loader.get_providers()
    families = set()

    for provider_config in providers.values():
        for model_config in provider_config.models.values():
            families.add(model_config.family)

    return families


def get_models_with_capability(capability: str) -> list[str]:
    """Get all models that support a specific capability.

    Args:
        capability: The capability name.

    Returns:
        List of model names.
    """
    return get_global_loader().get_models_by_capability(capability)


def supports_feature_dynamic(model_name: str, feature: str) -> bool:
    """Check if a model supports a feature using dynamic config.

    Args:
        model_name: The model name.
        feature: The feature name.

    Returns:
        True if the model supports the feature.
    """
    model_config = get_global_loader().get_model_config(model_name)
    if not model_config:
        return False

    feature_attr = f"supports_{feature}"
    if hasattr(model_config, feature_attr):
        return getattr(model_config, feature_attr)

    return False
