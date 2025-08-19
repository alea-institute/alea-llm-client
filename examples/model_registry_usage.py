"""Example usage of the model registry.

This example demonstrates how to use the comprehensive model registry
to discover and work with available models across all providers.
"""

from alea_llm_client.llms import (
    OpenAIModels,
    AnthropicModels,
    GoogleModels,
    GrokModels,
    get_all_models,
    get_provider_models,
    get_model_capabilities,
    supports_feature,
    DEFAULT_MODELS,
    MODEL_SIZES,
)
from alea_llm_client import OpenAIModel, AnthropicModel


def main():
    """Demonstrate model registry usage."""
    
    # 1. Using enum values for type-safe model selection
    print("=== Using Model Enums ===")
    
    # Create models using enum values
    openai_model = OpenAIModel(model=OpenAIModels.GPT_5_MINI.value)
    print(f"Created OpenAI model: {openai_model.model}")
    
    anthropic_model = AnthropicModel(model=AnthropicModels.CLAUDE_SONNET_4_20250514.value)
    print(f"Created Anthropic model: {anthropic_model.model}")
    
    # 2. Discovering available models
    print("\n=== Discovering Models ===")
    
    # Get all OpenAI models
    openai_models = get_provider_models("openai")
    print(f"Total OpenAI models available: {len(openai_models)}")
    
    # Show GPT-5 models only
    gpt5_models = [m for m in openai_models if m.startswith("gpt-5")]
    print(f"GPT-5 models: {gpt5_models[:3]}...")  # Show first 3
    
    # Show O-series reasoning models
    o_series = [m for m in openai_models if m.startswith(("o1", "o3", "o4"))]
    print(f"O-series reasoning models: {o_series[:3]}...")
    
    # 3. Checking model capabilities
    print("\n=== Model Capabilities ===")
    
    # Check if models support specific features
    test_models = [
        "gpt-5",
        "o3-mini",
        "claude-sonnet-4-20250514",
        "gemini-2.5-flash",
    ]
    
    for model in test_models:
        print(f"\n{model}:")
        print(f"  - Supports grammar: {supports_feature(model, 'grammar')}")
        print(f"  - Supports reasoning_effort: {supports_feature(model, 'reasoning_effort')}")
        print(f"  - Supports thinking: {supports_feature(model, 'thinking')}")
    
    # 4. Get detailed capabilities
    print("\n=== Detailed Capabilities ===")
    caps = get_model_capabilities("o3-mini")
    if caps:
        print(f"O3-mini capabilities:")
        print(f"  - Provider: {caps.provider}")
        print(f"  - Family: {caps.family}")
        print(f"  - Size: {caps.size}")
        print(f"  - Requires max_completion_tokens: {caps.requires_max_completion_tokens}")
    
    # 5. Using default models
    print("\n=== Default Models ===")
    for provider, model in DEFAULT_MODELS.items():
        print(f"{provider}: {model}")
    
    # 6. Finding models by size
    print("\n=== Models by Size ===")
    print(f"Mini models: {[m.value for m in MODEL_SIZES['mini'][:3]]}...")
    print(f"Flash models: {[m.value for m in MODEL_SIZES['flash']]}")
    print(f"Pro models: {[m.value for m in MODEL_SIZES['pro'][:3]]}...")
    
    # 7. Finding all models across providers
    print("\n=== Total Models ===")
    all_models = get_all_models()
    print(f"Total models across all providers: {len(all_models)}")
    
    # Count by provider
    for provider in ["openai", "anthropic", "google", "xai"]:
        provider_models = get_provider_models(provider)
        print(f"  {provider}: {len(provider_models)} models")


if __name__ == "__main__":
    main()