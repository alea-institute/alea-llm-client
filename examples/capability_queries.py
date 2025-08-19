"""Demonstration of model capability queries.

This example shows how to use the comprehensive capability matrix
to find models that meet specific requirements.
"""

from alea_llm_client.llms import (
    get_models_with_context_window_gte,
    get_models_with_capability,
    get_models_by_tier,
    filter_models,
    compare_models,
    generate_capability_table,
    get_model_details,
    ModelTier,
    ModelStatus,
)


def main():
    """Demonstrate capability queries."""
    
    print("=" * 80)
    print("Model Capability Query Examples")
    print("=" * 80)
    
    # 1. Find models with large context windows
    print("\n1. Models with context window >= 500,000 tokens:")
    large_context_models = get_models_with_context_window_gte(500000)
    for model in sorted(large_context_models)[:10]:  # Show first 10
        details = get_model_details(model)
        print(f"   - {model}: {details.context_window:,} tokens")
    
    # 2. Find models that support tools/function calling
    print("\n2. Models that support tool use / function calling:")
    tool_models = get_models_with_capability("tools")
    print(f"   Found {len(tool_models)} models with tool support")
    # Group by provider
    by_provider = {}
    for model in tool_models:
        details = get_model_details(model)
        if details:
            provider = details.provider
            if provider not in by_provider:
                by_provider[provider] = []
            by_provider[provider].append(model)
    
    for provider, models in by_provider.items():
        print(f"   {provider}: {len(models)} models")
    
    # 3. Find models with structured output (JSON mode)
    print("\n3. Models with structured output support:")
    structured_models = get_models_with_capability("structured_output")
    print(f"   Found {len(structured_models)} models")
    # Show a few examples
    for model in sorted(structured_models)[:5]:
        details = get_model_details(model)
        print(f"   - {model} ({details.provider})")
    
    # 4. Advanced filtering - find efficient models for specific use case
    print("\n4. Finding efficient models for API use:")
    print("   Requirements: context >= 100k, supports tools, not deprecated")
    
    efficient_models = filter_models(
        min_context=100000,
        capabilities=["tools", "structured_output"],
        tiers=[ModelTier.MINI, ModelTier.FLASH, ModelTier.NANO],
        exclude_deprecated=True
    )
    
    for model in sorted(efficient_models):
        details = get_model_details(model)
        print(f"   - {model}: {details.context_window:,} tokens, tier: {details.tier.value}")
    
    # 5. Find models with reasoning capabilities
    print("\n5. Models with advanced reasoning:")
    
    # O-series style reasoning
    reasoning_models = get_models_with_capability("reasoning")
    print(f"   O-series style reasoning: {len(reasoning_models)} models")
    for model in sorted(reasoning_models)[:5]:
        print(f"      - {model}")
    
    # Claude/Gemini style thinking
    thinking_models = get_models_with_capability("thinking")
    print(f"   Thinking capability: {len(thinking_models)} models")
    for model in sorted(thinking_models)[:5]:
        print(f"      - {model}")
    
    # 6. Compare specific models
    print("\n6. Comparing flagship models:")
    flagship_models = [
        "gpt-5",
        "claude-sonnet-4-20250514",
        "gemini-2.5-pro",
        "grok-4",
    ]
    
    comparison = compare_models(flagship_models)
    print("   " + "-" * 70)
    print(f"   {'Model':<30} {'Context':<12} {'Output':<10} {'Tools':<6} {'Vision':<7} {'Think':<6}")
    print("   " + "-" * 70)
    
    for model, caps in comparison.items():
        print(f"   {model:<30} {caps['context_window']:>10,}  {caps['max_output']:>8,}  "
              f"{'Yes' if caps['tools'] else 'No':<6} "
              f"{'Yes' if caps['vision'] else 'No':<7} "
              f"{'Yes' if caps['thinking'] else 'No':<6}")
    
    # 7. Find models by tier
    print("\n7. Models by performance tier:")
    for tier in [ModelTier.NANO, ModelTier.MINI, ModelTier.FLASH, ModelTier.PRO]:
        tier_models = get_models_by_tier(tier)
        print(f"   {tier.value}: {len(tier_models)} models")
    
    # 8. Multi-modal models
    print("\n8. Multi-modal models (vision + audio):")
    
    # Find models with vision
    vision_models = get_models_with_capability("vision")
    
    # Find models with audio
    audio_models = get_models_with_capability("audio")
    
    # Find intersection (both vision and audio)
    multimodal = []
    for model in vision_models:
        details = get_model_details(model)
        if details and (details.supports_audio_input or details.supports_audio_output):
            multimodal.append(model)
    
    print(f"   Found {len(multimodal)} truly multi-modal models:")
    for model in sorted(multimodal)[:5]:
        details = get_model_details(model)
        modalities = details.input_modalities + details.output_modalities
        unique_modalities = set(modalities)
        print(f"   - {model}: {', '.join(unique_modalities)}")
    
    # 9. Latest vs Legacy models
    print("\n9. Model status distribution:")
    status_count = {}
    for model_name in get_models_with_capability("tools"):  # Use any subset
        details = get_model_details(model_name)
        if details:
            status = details.status.value
            status_count[status] = status_count.get(status, 0) + 1
    
    for status, count in sorted(status_count.items()):
        print(f"   {status}: {count} models")
    
    # 10. Complex query - best model for specific use case
    print("\n10. Best models for production chatbot:")
    print("    Requirements: 100k+ context, tools, structured output, GA status")
    
    production_models = filter_models(
        min_context=100000,
        capabilities=["tools", "structured_output"],
        exclude_deprecated=True
    )
    
    # Filter further for GA status and good output size
    best_models = []
    for model in production_models:
        details = get_model_details(model)
        if details and details.status == ModelStatus.GA:
            output_tokens = details.max_completion_tokens or details.max_output_tokens
            if output_tokens and output_tokens >= 4096:
                best_models.append((model, details))
    
    # Sort by context window size
    best_models.sort(key=lambda x: x[1].context_window, reverse=True)
    
    print("    Top recommendations:")
    for model, details in best_models[:5]:
        output_tokens = details.max_completion_tokens or details.max_output_tokens
        print(f"    - {model}")
        print(f"      Provider: {details.provider}, Tier: {details.tier.value}")
        print(f"      Context: {details.context_window:,}, Output: {output_tokens:,}")
        print(f"      Pricing tier: {details.pricing_tier or 'standard'}")


if __name__ == "__main__":
    main()