"""Example usage of the model registry with 97 real models.

This example demonstrates how to use the comprehensive model registry
to discover and work with available models across all 4 providers.
Shows practical queries and capabilities of the real model data.
"""

from alea_llm_client import AnthropicModel, OpenAIModel
from alea_llm_client.llms import (
    DEFAULT_MODELS,
    MODEL_SIZES,
    get_all_models,
    get_model_capabilities,
    get_provider_models,
    supports_feature,
)


def main():
    """Demonstrate model registry usage with 97 real models."""

    print("🚀 ALEA LLM Client - 97 Real Models Available\n")

    # 1. Using dynamic model discovery for model selection
    print("=== 1. Dynamic Model Selection ===")

    # Create models using discovered model names (no actual API calls, just initialization)
    openai_models = get_provider_models("openai")
    gpt5_mini = next((m for m in openai_models if m == "gpt-5-mini"), "gpt-5-mini")

    openai_model = OpenAIModel(model=gpt5_mini)
    print(f"✓ OpenAI model initialized: {openai_model.model}")

    anthropic_models = get_provider_models("anthropic")
    claude_sonnet_4 = next((m for m in anthropic_models if "claude-sonnet-4-20250514" in m), "claude-sonnet-4-20250514")

    anthropic_model = AnthropicModel(model=claude_sonnet_4)
    print(f"✓ Anthropic model initialized: {anthropic_model.model}")

    # Show available GPT-5 models discovered from API
    available_gpt5_models = [m for m in openai_models if m.startswith("gpt-5")]
    print(f"✓ GPT-5 model options: {available_gpt5_models[:4]}... ({len(available_gpt5_models)} total)")

    # 2. Discovering available models across all providers
    print("\n=== 2. Model Discovery ===")

    # Get total count first
    all_models = get_all_models()
    print(f"🎯 Total models across all providers: {len(all_models)}")

    # Break down by provider
    providers = ["openai", "anthropic", "google", "xai"]
    for provider in providers:
        provider_models = get_provider_models(provider)
        print(f"  📊 {provider.capitalize()}: {len(provider_models)} models")

    # Show specific model families
    openai_models = get_provider_models("openai")

    # GPT-5 models (latest generation with grammar support)
    gpt5_models = sorted([m for m in openai_models if m.startswith("gpt-5")])
    print(f"\n🔥 GPT-5 models ({len(gpt5_models)}): {gpt5_models[:4]}...")

    # O-series reasoning models (require max_completion_tokens)
    o_series = sorted([m for m in openai_models if m.startswith(("o1", "o3", "o4"))])
    print(f"🧠 O-series reasoning models ({len(o_series)}): {o_series[:4]}...")

    # Show Claude 4 models
    anthropic_models = get_provider_models("anthropic")
    claude4_models = sorted(
        [
            m
            for m in anthropic_models
            if "claude-4" in m or "claude-sonnet-4" in m or "claude-opus-4" in m
        ]
    )
    print(f"🎭 Claude 4 models ({len(claude4_models)}): {claude4_models[:3]}...")

    # Show Gemini 2.5 models (latest with thinking)
    google_models = get_provider_models("google")
    gemini25_models = sorted([m for m in google_models if "gemini-2.5" in m])
    print(f"💎 Gemini 2.5 models ({len(gemini25_models)}): {gemini25_models}")

    # 3. Checking model capabilities with real data
    print("\n=== 3. Model Capabilities Analysis ===")

    # Test flagship models from each provider
    flagship_models = [
        "gpt-5",  # OpenAI's latest with grammar
        "o3-mini",  # OpenAI reasoning model
        "claude-sonnet-4-20250514",  # Anthropic's latest with thinking
        "gemini-2.5-pro",  # Google's latest with thinking
        "grok-4-0709",  # xAI's flagship
    ]

    print("🔍 Flagship Model Feature Comparison:")
    for model in flagship_models:
        print(f"\n📋 {model}:")
        features = [
            ("grammar", "Grammar constraints"),
            ("reasoning_effort", "Reasoning effort control"),
            ("thinking", "Thinking/reasoning mode"),
            ("tools", "Function calling"),
            ("vision", "Image understanding"),
            ("structured_output", "JSON mode"),
            ("audio", "Audio capabilities"),
            ("search", "Web search"),
        ]

        supported = []
        for feature_key, feature_name in features:
            if supports_feature(model, feature_key):
                supported.append(feature_name)

        if supported:
            print(f"    ✅ Supports: {', '.join(supported)}")
        else:
            print("    ⚠️  Limited feature set or capabilities not detected")

    # 4. Detailed model analysis
    print("\n=== 4. Detailed Model Analysis ===")

    # Analyze O-series reasoning requirements
    o_mini = get_model_capabilities("o3-mini")
    if o_mini:
        print("🧠 O3-mini detailed analysis:")
        print(f"    Provider: {o_mini.provider}")
        print(f"    Family: {o_mini.family}")
        print(f"    Tier: {o_mini.size}")
        print(
            f"    ⚠️  Requires max_completion_tokens: {o_mini.requires_max_completion_tokens}"
        )
        print(f"    Reasoning effort: {o_mini.supports_reasoning_effort}")

    # Analyze GPT-5 grammar feature
    gpt5 = get_model_capabilities("gpt-5")
    if gpt5:
        print("\n🔥 GPT-5 analysis:")
        print(f"    Grammar support: {gpt5.supports_grammar} (unique to GPT-5 family)")
        print(f"    Vision support: {gpt5.supports_vision}")
        print(f"    Tools support: {supports_feature('gpt-5', 'tools')}")

    # Analyze Claude 4 thinking capabilities
    claude4 = get_model_capabilities("claude-sonnet-4-20250514")
    if claude4:
        print("\n🎭 Claude-Sonnet-4 analysis:")
        print(
            f"    Thinking mode: {claude4.supports_thinking} (Claude's reasoning approach)"
        )
        print(f"    Vision support: {claude4.supports_vision}")
        print(f"    Family: {claude4.family}")

    # 5. Provider defaults and recommendations
    print("\n=== 5. Provider Defaults & Recommendations ===")
    print("🎯 Default models per provider (automatically selected for best balance):")
    for provider, model in DEFAULT_MODELS.items():
        caps = get_model_capabilities(model)
        context = "Unknown context"
        if caps:
            # Get detailed capabilities for context window info
            from alea_llm_client.llms.model_capabilities import get_model_details

            details = get_model_details(model)
            if details:
                context = f"{details.context_window:,} tokens"

        print(f"  📌 {provider.capitalize()}: {model} ({context})")

    # 6. Finding models by performance tier
    print("\n=== 6. Models by Performance Tier ===")
    print("📊 Model distribution by performance tier:")

    tier_counts = {}
    for tier_name in [
        "nano",
        "mini",
        "small",
        "flash",
        "medium",
        "standard",
        "large",
        "pro",
        "heavy",
    ]:
        tier_models = MODEL_SIZES.get(tier_name, [])
        if tier_models:
            tier_counts[tier_name] = len(tier_models)
            # Show a few examples for key tiers
            if tier_name in ["mini", "flash", "pro"]:
                examples = tier_models[:3] if len(tier_models) > 3 else tier_models
                print(
                    f"  {tier_name.upper()}: {len(tier_models)} models - Examples: {examples}"
                )
            else:
                print(f"  {tier_name.upper()}: {len(tier_models)} models")

    print(
        "\n💡 Tier guide: nano < mini < small/flash < medium/standard < large/pro < heavy"
    )
    print("💰 Generally: nano/mini = cheapest, pro/heavy = most capable")

    # 7. Advanced queries and real-world usage patterns
    print("\n=== 7. Advanced Model Queries ===")

    # Find models with large context windows (real data)
    print("🔍 Advanced filtering examples:")

    # Models with 500K+ context (great for document analysis)
    from alea_llm_client.llms.model_capabilities import (
        get_models_with_context_window_gte,
    )

    large_context = get_models_with_context_window_gte(500000)
    print(f"  📄 Models with 500K+ context: {len(large_context)} models")
    print(f"      Examples: {sorted(large_context)[:4]}...")

    # Multi-modal models (vision + tools)
    vision_models = set()
    tool_models = set()
    for model in all_models:
        if supports_feature(model, "vision"):
            vision_models.add(model)
        if supports_feature(model, "tools"):
            tool_models.add(model)

    multimodal_tools = vision_models & tool_models
    print(f"  👁️  Vision + Tools capable: {len(multimodal_tools)} models")
    print(f"      Examples: {sorted(multimodal_tools)[:3]}...")

    # Find reasoning models
    reasoning_models = set()
    thinking_models = set()
    for model in all_models:
        if supports_feature(model, "reasoning_effort"):
            reasoning_models.add(model)
        if supports_feature(model, "thinking"):
            thinking_models.add(model)

    print(f"  🧠 O-series reasoning: {len(reasoning_models)} models")
    print(f"  💭 Thinking mode: {len(thinking_models)} models")

    # Audio-capable models
    audio_models = {model for model in all_models if supports_feature(model, "audio")}
    print(f"  🔊 Audio capable: {len(audio_models)} models")
    if audio_models:
        print(f"      Examples: {sorted(audio_models)[:3]}...")
    # 8. Practical recommendations
    print("\n=== 8. Practical Usage Recommendations ===")
    print("💡 Model selection guide based on real capabilities:")
    print("")
    print("  🚀 For general chat/assistance:")
    print("    • gpt-5 (grammar constraints, latest features)")
    print("    • claude-sonnet-4-20250514 (thinking mode, large context)")
    print("    • gemini-2.5-pro (multimodal, thinking)")
    print("")
    print("  🧠 For complex reasoning:")
    print("    • o3-mini (cost-effective reasoning)")
    print("    • o1 or o3 (maximum reasoning capability)")
    print("    • claude-opus-4-20250514 (thinking mode)")
    print("")
    print("  💰 For cost-effective tasks:")
    print("    • gpt-5-nano (fast, cheap, grammar support)")
    print("    • claude-3-5-haiku-20241022 (fast, vision)")
    print("    • gemini-2.0-flash (multimodal, fast)")
    print("")
    print("  🎨 For multimodal tasks:")
    print("    • gpt-4o-2024-11-20 (vision, tools)")
    print("    • claude-sonnet-4-20250514 (vision, thinking)")
    print("    • gemini-2.5-pro (vision, video, audio)")
    print("")
    print("  🔊 For audio/speech:")
    if audio_models:
        audio_examples = sorted(audio_models)[:3]
        print(f"    • {', '.join(audio_examples)}")
    else:
        print("    • Check for models with 'audio' in the name")
    print("")
    print(f"✨ Total of {len(all_models)} real models available across 4 providers!")
    print("🔑 All models use API keys from ~/.alea/keys/ (no hardcoding needed)")


if __name__ == "__main__":
    main()
