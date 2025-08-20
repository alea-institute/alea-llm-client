"""Advanced model capability queries with 97 real models.

This example shows how to use the comprehensive capability matrix
to find models that meet specific requirements. All data is from
real API documentation and live model availability.
"""

from alea_llm_client.llms import (
    ModelStatus,
    ModelTier,
    compare_models,
    filter_models,
    get_all_models,
    get_model_details,
    get_models_by_tier,
    get_models_with_capability,
    get_models_with_context_window_gte,
)


def main():
    """Demonstrate advanced capability queries with real model data."""

    print("🔍" + "=" * 78)
    print("  ALEA LLM Client - Advanced Model Capability Queries (97 Models)")
    print("=" * 80)
    print("📈 Query real models across OpenAI, Anthropic, Google, and xAI")
    print("🚀 All data sourced from official SDKs and API documentation\n")

    # 1. Find models with large context windows (document analysis)
    print("1. 📄 Models with context window >= 500,000 tokens:")
    print("   Perfect for document analysis, code review, and long conversations")

    large_context_models = get_models_with_context_window_gte(500000)
    print(f"   Found {len(large_context_models)} models with 500K+ context")

    # Group by context window size for better understanding
    context_groups = {}
    for model in large_context_models:
        details = get_model_details(model)
        if details:
            window = details.context_window
            if window not in context_groups:
                context_groups[window] = []
            context_groups[window].append((model, details))

    # Show the largest context windows first
    for window in sorted(context_groups.keys(), reverse=True):
        models = context_groups[window]
        if window >= 1000000:  # 1M+ tokens
            print(f"\n   🔥 {window:,} token models ({len(models)}):")
            for model, details in sorted(models)[:5]:
                print(f"      • {model} ({details.provider})")
        elif len(models) <= 3:  # Show all if just a few
            print(f"\n   📋 {window:,} token models:")
            for model, details in sorted(models):
                print(f"      • {model} ({details.provider})")

    # 2. Find models that support tools/function calling
    print("\n2. 🛠️  Models that support tool use / function calling:")
    print("   Essential for building AI agents, API integrations, and automation")

    tool_models = get_models_with_capability("tools")
    print(f"   ✓ Found {len(tool_models)} models with tool support")

    # Group by provider and show distribution
    by_provider = {}
    for model in tool_models:
        details = get_model_details(model)
        if details:
            provider = details.provider
            if provider not in by_provider:
                by_provider[provider] = []
            by_provider[provider].append((model, details))

    for provider in sorted(by_provider.keys()):
        models = by_provider[provider]
        print(f"   📌 {provider.capitalize()}: {len(models)} models")

        # Show a few examples of different tiers
        tiers_shown = set()
        for model, details in sorted(models):
            if details.tier.value not in tiers_shown and len(tiers_shown) < 3:
                print(f"      • {model} ({details.tier.value} tier)")
                tiers_shown.add(details.tier.value)

    # 3. Find models with structured output (JSON mode)
    print("\n3. 📋 Models with structured output support:")
    print(
        "   Critical for building reliable APIs, data extraction, and structured responses"
    )

    structured_models = get_models_with_capability("structured_output")
    print(f"   ✓ Found {len(structured_models)} models with JSON mode")

    # Group by provider and show best examples
    structured_by_provider = {}
    for model in structured_models:
        details = get_model_details(model)
        if details:
            provider = details.provider
            if provider not in structured_by_provider:
                structured_by_provider[provider] = []
            structured_by_provider[provider].append((model, details))

    for provider in sorted(structured_by_provider.keys()):
        models = structured_by_provider[provider]
        print(f"   🔸 {provider.capitalize()}: {len(models)} models")

        # Show flagship model for structured output from each provider
        flagship_models = []
        for model, details in models:
            if (
                (provider == "openai" and ("gpt-5" in model or "o3" in model))
                or (provider == "anthropic" and "claude-4" in model)
                or (provider == "google" and "gemini-2.5" in model)
                or (provider == "xai" and "grok-4" in model)
            ):
                flagship_models.append((model, details))

        if flagship_models:
            for model, details in flagship_models[:2]:  # Show top 2
                print(f"      • {model} ({details.tier.value} tier)")

    # 4. Advanced filtering - find efficient models for production API use
    print("\n4. 🚀 Finding efficient models for production API use:")
    print(
        "   Requirements: 100K+ context, tools + JSON, cost-effective tiers, not deprecated"
    )

    efficient_models = filter_models(
        min_context=100000,
        capabilities=["tools", "structured_output"],
        tiers=[ModelTier.MINI, ModelTier.FLASH, ModelTier.NANO, ModelTier.MEDIUM],
        exclude_deprecated=True,
    )

    print(f"   ✓ Found {len(efficient_models)} production-ready models")

    # Group by provider and show best options
    efficient_by_provider = {}
    for model in efficient_models:
        details = get_model_details(model)
        if details:
            provider = details.provider
            if provider not in efficient_by_provider:
                efficient_by_provider[provider] = []
            efficient_by_provider[provider].append((model, details))

    for provider in sorted(efficient_by_provider.keys()):
        models = efficient_by_provider[provider]
        print(f"\n   📈 {provider.capitalize()} recommendations:")

        # Sort by context window size (bigger is often better for production)
        models_sorted = sorted(models, key=lambda x: x[1].context_window, reverse=True)

        for model, details in models_sorted[:3]:  # Top 3 per provider
            pricing_info = f", {details.pricing_tier}" if details.pricing_tier else ""
            print(f"      • {model}")
            print(
                f"        {details.context_window:,} tokens, {details.tier.value} tier{pricing_info}"
            )

    # 5. Find models with advanced reasoning capabilities
    print("\n5. 🧠 Models with advanced reasoning:")
    print(
        "   Two approaches: OpenAI O-series deep reasoning vs Claude/Gemini thinking mode"
    )

    # O-series style reasoning (requires max_completion_tokens)
    reasoning_models = get_models_with_capability("reasoning")
    print(f"\n   🔍 O-series deep reasoning: {len(reasoning_models)} models")
    print("   ⚠️  Note: These require max_completion_tokens parameter")

    # Group by model family
    o_families = {}
    for model in reasoning_models:
        details = get_model_details(model)
        if details:
            family = details.family
            if family not in o_families:
                o_families[family] = []
            o_families[family].append((model, details))

    for family in sorted(o_families.keys()):
        models = o_families[family]
        print(f"      📊 {family}: {len(models)} variants")
        # Show different tiers
        for model, details in sorted(models)[:3]:
            max_tokens = details.max_completion_tokens or details.max_output_tokens
            print(
                f"         • {model} ({details.tier.value}, {max_tokens:,} max tokens)"
            )

    # Claude/Gemini style thinking
    thinking_models = get_models_with_capability("thinking")
    print(f"\n   💭 Thinking/reasoning mode: {len(thinking_models)} models")
    print("   ✨ These models show their reasoning process in responses")

    thinking_by_provider = {}
    for model in thinking_models:
        details = get_model_details(model)
        if details:
            provider = details.provider
            if provider not in thinking_by_provider:
                thinking_by_provider[provider] = []
            thinking_by_provider[provider].append((model, details))

    for provider in sorted(thinking_by_provider.keys()):
        models = thinking_by_provider[provider]
        print(f"      🎭️ {provider.capitalize()}: {len(models)} models")
        # Show latest/best models
        latest_models = []
        for model, details in models:
            if (
                provider == "anthropic"
                and ("claude-4" in model or "claude-3-7" in model)
            ) or (provider == "google" and "gemini-2.5" in model):
                latest_models.append((model, details))

        for model, details in latest_models[:3]:
            print(f"         • {model} ({details.context_window:,} context)")

    # 6. Compare flagship models across providers
    print("\n6. 🏆 Flagship model comparison:")
    print("   Head-to-head comparison of the best from each provider")

    flagship_models = [
        "gpt-5",  # OpenAI latest with grammar
        "claude-sonnet-4-20250514",  # Anthropic latest with thinking
        "gemini-2.5-pro",  # Google latest with thinking
        "grok-4-0709",  # xAI flagship
    ]

    comparison = compare_models(flagship_models)

    # Enhanced comparison table
    print("\n   " + "=" * 85)
    header = f"   {'Model':<32} {'Context':<10} {'Output':<8} {'Tools':<6} {'Vision':<7} {'Audio':<6} {'Special':<12}"
    print(header)
    print("   " + "=" * 85)

    for model in flagship_models:
        if model in comparison:
            caps = comparison[model]
            details = get_model_details(model)

            # Determine special capability
            special = ""
            if details:
                if details.supports_grammar:
                    special = "Grammar"
                elif details.supports_thinking:
                    special = "Thinking"
                elif details.supports_reasoning:
                    special = "Reasoning"
                elif details.supports_search:
                    special = "Search"

            context_str = (
                f"{caps['context_window']:,}"
                if caps["context_window"] < 1000000
                else f"{caps['context_window'] // 1000000}M"
            )
            output_str = (
                f"{caps['max_output']:,}"
                if caps["max_output"] < 1000000
                else f"{caps['max_output'] // 1000000}M"
            )

            print(
                f"   {model:<32} {context_str:>9} {output_str:>7} "
                f"{'✓' if caps['tools'] else '✗':<6} "
                f"{'✓' if caps['vision'] else '✗':<7} "
                f"{'✓' if caps.get('audio', False) else '✗':<6} "
                f"{special:<12}"
            )

    print("   " + "=" * 85)

    # 7. Models by performance tier and cost optimization
    print("\n7. 💰 Models by performance tier (cost optimization):")
    print("   Understanding the performance/cost spectrum")

    tier_analysis = {}
    for tier in [
        ModelTier.NANO,
        ModelTier.MINI,
        ModelTier.SMALL,
        ModelTier.FLASH,
        ModelTier.MEDIUM,
        ModelTier.STANDARD,
        ModelTier.LARGE,
        ModelTier.PRO,
        ModelTier.HEAVY,
    ]:
        tier_models = get_models_by_tier(tier)
        if tier_models:
            tier_analysis[tier] = tier_models

    print("\n   Performance tiers (from most cost-effective to most capable):")

    tier_order = [
        ModelTier.NANO,
        ModelTier.MINI,
        ModelTier.SMALL,
        ModelTier.FLASH,
        ModelTier.MEDIUM,
        ModelTier.STANDARD,
        ModelTier.LARGE,
        ModelTier.PRO,
        ModelTier.HEAVY,
    ]

    for tier in tier_order:
        if tier in tier_analysis:
            models = tier_analysis[tier]
            print(f"\n   📊 {tier.value.upper()}: {len(models)} models")

            # Show provider distribution for this tier
            provider_count = {}
            example_models = []

            for model in models:
                details = get_model_details(model)
                if details:
                    provider = details.provider
                    provider_count[provider] = provider_count.get(provider, 0) + 1

                    # Collect interesting examples
                    if len(example_models) < 3:
                        example_models.append(model)

            # Show provider distribution
            provider_summary = ", ".join(
                [f"{p}: {c}" for p, c in sorted(provider_count.items())]
            )
            print(f"      Providers: {provider_summary}")

            # Show examples
            if example_models:
                print(f"      Examples: {', '.join(example_models)}")

    print("\n   💡 Tier selection guide:")
    print("      NANO/MINI: High-volume, simple tasks, cost-sensitive")
    print("      FLASH/SMALL: Balanced performance and cost")
    print("      STANDARD/MEDIUM: General-purpose, good capabilities")
    print("      LARGE/PRO/HEAVY: Complex tasks, maximum capabilities")

    # 8. Multi-modal capabilities analysis
    print("\n8. 🎨 Multi-modal capabilities analysis:")
    print("   Understanding which models handle different types of content")

    # Find models with vision
    vision_models = get_models_with_capability("vision")
    print(f"\n   👁️  Vision-capable models: {len(vision_models)}")

    # Find models with audio
    audio_models = get_models_with_capability("audio")
    print(f"   🔊 Audio-capable models: {len(audio_models)}")

    # Find truly multi-modal (vision + audio)
    multimodal = []
    for model in vision_models:
        details = get_model_details(model)
        if details and (details.supports_audio_input or details.supports_audio_output):
            multimodal.append((model, details))

    print(f"   🌟 Truly multi-modal (vision + audio): {len(multimodal)}")

    if multimodal:
        print("\n   Top multi-modal models:")
        for model, details in sorted(multimodal)[:5]:
            input_modalities = details.input_modalities or ["text"]
            output_modalities = details.output_modalities or ["text"]

            input_str = ", ".join(input_modalities)
            output_str = ", ".join(output_modalities)

            print(f"      • {model} ({details.provider})")
            print(f"        Input: {input_str}")
            print(f"        Output: {output_str}")

    # Analyze vision-only models by provider
    vision_by_provider = {}
    for model in vision_models:
        details = get_model_details(model)
        if details:
            provider = details.provider
            if provider not in vision_by_provider:
                vision_by_provider[provider] = []
            vision_by_provider[provider].append(model)

    print("\n   Vision capability by provider:")
    for provider in sorted(vision_by_provider.keys()):
        models = vision_by_provider[provider]
        print(f"      📷 {provider.capitalize()}: {len(models)} vision-capable models")

        # Show best vision model from each provider
        best_model = None
        for model in models:
            details = get_model_details(model)
            if details and (
                (provider == "openai" and "gpt-4o" in model)
                or (provider == "anthropic" and "claude-4" in model)
                or (provider == "google" and "gemini-2" in model)
                or (provider == "xai" and "vision" in model)
            ):
                best_model = model
                break

        if best_model:
            print(f"         Best: {best_model}")

    # 9. Model lifecycle and status analysis
    print("\n9. 🗓️ Model lifecycle and status analysis:")
    print("   Understanding which models are current vs deprecated")

    status_count = {}
    status_by_provider = {}

    all_models_list = get_all_models()
    for model_name in all_models_list:
        details = get_model_details(model_name)
        if details:
            status = details.status.value
            provider = details.provider

            status_count[status] = status_count.get(status, 0) + 1

            if provider not in status_by_provider:
                status_by_provider[provider] = {}
            status_by_provider[provider][status] = (
                status_by_provider[provider].get(status, 0) + 1
            )

    print(f"\n   Overall status distribution ({len(all_models_list)} total models):")
    for status in ["ga", "preview", "beta", "experimental", "legacy", "deprecated"]:
        if status in status_count:
            count = status_count[status]
            percentage = (count / len(all_models_list)) * 100
            status_emoji = {
                "ga": "✅",
                "preview": "🔍",
                "beta": "⚠️",
                "experimental": "🧪",
                "legacy": "📦",
                "deprecated": "❌",
            }.get(status, "💬")

            print(
                f"      {status_emoji} {status.upper()}: {count} models ({percentage:.1f}%)"
            )

    print("\n   Status by provider:")
    for provider in sorted(status_by_provider.keys()):
        provider_statuses = status_by_provider[provider]
        total = sum(provider_statuses.values())
        ga_count = provider_statuses.get("ga", 0)
        preview_count = provider_statuses.get("preview", 0)
        legacy_count = provider_statuses.get("legacy", 0)

        print(f"      📊 {provider.capitalize()}: {total} total")
        print(
            f"         GA: {ga_count}, Preview: {preview_count}, Legacy: {legacy_count}"
        )

    # 10. Real-world use case recommendations
    print("\n10. 🎯 Real-world use case recommendations:")
    print("    Finding the best models for specific applications")

    # Use case 1: Production chatbot
    print("\n    🤖 Production Chatbot Requirements:")
    print("       100K+ context, tools, JSON output, GA status, good output size")

    production_models = filter_models(
        min_context=100000,
        capabilities=["tools", "structured_output"],
        exclude_deprecated=True,
    )

    # Filter for GA status and good output size
    chatbot_models = []
    for model in production_models:
        details = get_model_details(model)
        if details and details.status == ModelStatus.GA:
            output_tokens = details.max_completion_tokens or details.max_output_tokens
            if output_tokens and output_tokens >= 4096:
                chatbot_models.append((model, details))

    # Sort by context window size
    chatbot_models.sort(key=lambda x: x[1].context_window, reverse=True)

    print("       ✅ Top recommendations:")
    for model, details in chatbot_models[:4]:
        output_tokens = details.max_completion_tokens or details.max_output_tokens
        pricing = details.pricing_tier or "standard"
        print(f"          • {model} ({details.provider})")
        print(
            f"            {details.context_window:,} context, {output_tokens:,} output, {pricing} pricing"
        )

    # Use case 2: Document analysis
    print("\n    📄 Document Analysis Requirements:")
    print("       1M+ context, vision helpful, tools for extraction")

    doc_models = []
    for model in get_models_with_context_window_gte(1000000):
        details = get_model_details(model)
        if details:
            has_tools = get_models_with_capability("tools")
            if model in has_tools:
                doc_models.append((model, details))

    print("       ✅ Best for large documents:")
    for model, details in sorted(
        doc_models, key=lambda x: x[1].context_window, reverse=True
    )[:4]:
        vision_info = " + vision" if details.supports_vision else ""
        print(f"          • {model}{vision_info}")
        print(f"            {details.context_window:,} tokens ({details.provider})")

    # Use case 3: Creative/multimodal tasks
    print("\n    🎨 Creative/Multimodal Requirements:")
    print("       Vision, audio helpful, good output length, latest models")

    creative_models = []
    vision_models_set = set(get_models_with_capability("vision"))

    for model in vision_models_set:
        details = get_model_details(model)
        if details and details.status in [ModelStatus.GA, ModelStatus.PREVIEW]:
            output_tokens = details.max_completion_tokens or details.max_output_tokens
            if output_tokens and output_tokens >= 4096:
                creative_models.append((model, details))

    # Sort by recency/capability
    creative_models.sort(
        key=lambda x: (x[1].context_window, x[1].supports_audio_input), reverse=True
    )

    print("       ✅ Best creative models:")
    for model, details in creative_models[:4]:
        capabilities = []
        if details.supports_vision:
            capabilities.append("vision")
        if details.supports_audio_input or details.supports_audio_output:
            capabilities.append("audio")
        if details.supports_thinking:
            capabilities.append("thinking")

        cap_str = ", ".join(capabilities) if capabilities else "basic"
        print(f"          • {model} ({cap_str})")
        print(f"            {details.context_window:,} context ({details.provider})")

    print(
        "\n⭐ Analysis complete! Use these insights to choose the right model for your needs."
    )
    print("🔑 Remember: All models use API keys from ~/.alea/keys/ automatically.")


if __name__ == "__main__":
    main()
