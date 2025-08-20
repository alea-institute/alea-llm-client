#!/usr/bin/env python3
"""Generate a comprehensive model comparison table with 97 real models.

This script creates a detailed Markdown table showing all 97 models
across OpenAI, Anthropic, Google, and xAI with their real capabilities.
"""

from datetime import datetime

from alea_llm_client.llms import (
    COMPLETE_MODEL_CAPABILITIES,
)


def generate_detailed_table():
    """Generate a detailed comparison table with all 97 real models."""

    lines = []
    lines.append("# Complete Model Capability Matrix - 97 Real Models")
    lines.append("")
    lines.append("🚀 Generated from alea-llm-client model registry with real API data")
    lines.append("📊 Data sourced from official SDKs: OpenAI, Anthropic, Google, xAI")
    lines.append(f"📅 Last updated: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("")
    lines.append("## Quick Summary")
    lines.append("")

    # Add summary stats at the top
    provider_counts = {}
    for name, caps in COMPLETE_MODEL_CAPABILITIES.items():
        provider = caps.provider
        provider_counts[provider] = provider_counts.get(provider, 0) + 1

    lines.append(f"- **Total Models**: {len(COMPLETE_MODEL_CAPABILITIES)}")
    for provider, count in sorted(provider_counts.items()):
        lines.append(f"- **{provider.capitalize()}**: {count} models")
    lines.append("")

    # Create enhanced header with more relevant info
    lines.append("## Complete Model Matrix")
    lines.append("")
    lines.append(
        "| Provider | Model | Tier | Status | Context | Max Output | Tools | Vision | Audio | Reasoning | Thinking | Grammar | Search | Pricing |"
    )
    lines.append(
        "|----------|-------|------|--------|---------|------------|-------|--------|-------|-----------|----------|---------|--------|---------|"
    )

    # Group by provider
    by_provider = {}
    for name, caps in COMPLETE_MODEL_CAPABILITIES.items():
        provider = caps.provider
        if provider not in by_provider:
            by_provider[provider] = []
        by_provider[provider].append((name, caps))

    # Sort within each provider
    for provider in sorted(by_provider.keys()):
        models = sorted(by_provider[provider], key=lambda x: x[0])

        for name, caps in models:
            # Determine output tokens (handle both regular and completion tokens)
            output = caps.max_completion_tokens or caps.max_output_tokens

            # Format large numbers with K/M suffix
            def format_tokens(num):
                if num >= 1000000:
                    return f"{num // 1000000}M"
                elif num >= 1000:
                    return f"{num // 1000}K"
                else:
                    return str(num)

            context_str = format_tokens(caps.context_window)
            output_str = format_tokens(output)

            # Enhanced pricing info
            pricing = caps.pricing_tier or "standard"
            pricing_emoji = {"low": "💚", "medium": "🟡", "premium": "🔴"}.get(
                pricing, "⚪"
            )

            # Build the enhanced row
            row = [
                caps.provider,
                name,
                caps.tier.value,
                caps.status.value,
                context_str,
                output_str,
                "✅" if caps.supports_tools else "❌",
                "✅" if caps.supports_vision else "❌",
                "✅"
                if caps.supports_audio_input or caps.supports_audio_output
                else "❌",
                "✅" if caps.supports_reasoning else "❌",
                "✅" if caps.supports_thinking else "❌",
                "✅" if caps.supports_grammar else "❌",
                "✅" if caps.supports_search else "❌",
                f"{pricing_emoji} {pricing}",
            ]

            lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Legend")
    lines.append("")
    lines.append(
        "- **Tier**: nano < mini < small/flash < medium/standard < large/pro < heavy"
    )
    lines.append(
        "- **Status**: ga (Generally Available), preview, beta, experimental, legacy, deprecated"
    )
    lines.append("- **Context**: Maximum input tokens (K=thousands, M=millions)")
    lines.append("- **Max Output**: Maximum output tokens")
    lines.append("- **Tools**: Supports function calling/tool use")
    lines.append("- **Vision**: Supports image input")
    lines.append("- **Audio**: Supports audio input or output")
    lines.append("- **Reasoning**: O-series style deep reasoning")
    lines.append("- **Thinking**: Claude/Gemini style thinking mode")
    lines.append("- **Grammar**: Supports grammar constraints (GPT-5)")
    lines.append("- **Search**: Supports web search")

    return "\n".join(lines)


def generate_summary_stats():
    """Generate summary statistics about the model registry."""

    lines = []
    lines.append("## Model Registry Statistics")
    lines.append("")

    # Count by provider
    by_provider = {}
    for _name, caps in COMPLETE_MODEL_CAPABILITIES.items():
        provider = caps.provider
        by_provider[provider] = by_provider.get(provider, 0) + 1

    lines.append("### Models by Provider")
    for provider, count in sorted(by_provider.items()):
        lines.append(f"- {provider}: {count} models")

    lines.append(f"- **Total**: {len(COMPLETE_MODEL_CAPABILITIES)} models")
    lines.append("")

    # Count capabilities
    capability_counts = {
        "Tool/Function calling": sum(
            1 for c in COMPLETE_MODEL_CAPABILITIES.values() if c.supports_tools
        ),
        "Structured output (JSON)": sum(
            1
            for c in COMPLETE_MODEL_CAPABILITIES.values()
            if c.supports_structured_output
        ),
        "Vision/Image input": sum(
            1 for c in COMPLETE_MODEL_CAPABILITIES.values() if c.supports_vision
        ),
        "Audio capabilities": sum(
            1
            for c in COMPLETE_MODEL_CAPABILITIES.values()
            if c.supports_audio_input or c.supports_audio_output
        ),
        "Reasoning (O-series)": sum(
            1 for c in COMPLETE_MODEL_CAPABILITIES.values() if c.supports_reasoning
        ),
        "Thinking mode": sum(
            1 for c in COMPLETE_MODEL_CAPABILITIES.values() if c.supports_thinking
        ),
        "Grammar constraints": sum(
            1 for c in COMPLETE_MODEL_CAPABILITIES.values() if c.supports_grammar
        ),
        "Web search": sum(
            1 for c in COMPLETE_MODEL_CAPABILITIES.values() if c.supports_search
        ),
        "Fine-tuning": sum(
            1 for c in COMPLETE_MODEL_CAPABILITIES.values() if c.supports_fine_tuning
        ),
    }

    lines.append("### Models with Key Capabilities")
    for capability, count in capability_counts.items():
        percentage = (count / len(COMPLETE_MODEL_CAPABILITIES)) * 100
        lines.append(f"- {capability}: {count} models ({percentage:.1f}%)")

    lines.append("")

    # Context window ranges
    contexts = [caps.context_window for caps in COMPLETE_MODEL_CAPABILITIES.values()]
    lines.append("### Context Window Distribution")
    lines.append(f"- < 50K tokens: {sum(1 for c in contexts if c < 50000)} models")
    lines.append(
        f"- 50K-200K tokens: {sum(1 for c in contexts if 50000 <= c < 200000)} models"
    )
    lines.append(
        f"- 200K-1M tokens: {sum(1 for c in contexts if 200000 <= c < 1000000)} models"
    )
    lines.append(f"- ≥ 1M tokens: {sum(1 for c in contexts if c >= 1000000)} models")

    return "\n".join(lines)


def generate_footer():
    """Generate footer with usage instructions."""

    lines = []
    lines.append("## How to Use This Data")
    lines.append("")
    lines.append("### With ALEA LLM Client")
    lines.append("")
    lines.append("```python")
    lines.append("from alea_llm_client import OpenAIModel, AnthropicModel")
    lines.append(
        "from alea_llm_client.llms import get_model_capabilities, supports_feature"
    )
    lines.append("")
    lines.append("# Use any model from the table")
    lines.append("model = OpenAIModel(model='gpt-5')")
    lines.append("response = model.chat('Hello, world!')")
    lines.append("")
    lines.append("# Check capabilities programmatically")
    lines.append("caps = get_model_capabilities('claude-sonnet-4-20250514')")
    lines.append("has_vision = supports_feature('gpt-4o', 'vision')")
    lines.append("```")
    lines.append("")
    lines.append("### Model Selection Guide")
    lines.append("")
    lines.append("- **For cost-effective tasks**: Choose nano/mini tier models")
    lines.append(
        "- **For complex reasoning**: Use O-series or Claude-4/Gemini-2.5 with thinking"
    )
    lines.append(
        "- **For multimodal tasks**: Pick models with vision + audio capabilities"
    )
    lines.append(
        "- **For production APIs**: Prefer GA status models with tools + JSON support"
    )
    lines.append("- **For large documents**: Select models with 500K+ context windows")
    lines.append("")
    lines.append("### API Configuration")
    lines.append("")
    lines.append("All models automatically use API keys from:")
    lines.append("- OpenAI: `~/.alea/keys/openai`")
    lines.append("- Anthropic: `~/.alea/keys/anthropic`")
    lines.append("- Google: `~/.alea/keys/google`")
    lines.append("- xAI: `~/.alea/keys/xai`")
    lines.append("")
    lines.append("---")
    lines.append(
        f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} by alea-llm-client*"
    )

    return "\n".join(lines)


def main():
    """Generate and save the comprehensive comparison tables."""

    print("🚀 Generating comprehensive model comparison table...")
    print(f"📊 Processing {len(COMPLETE_MODEL_CAPABILITIES)} models across 4 providers")

    # Generate the detailed table
    print("   📋 Building detailed capability matrix...")
    detailed_table = generate_detailed_table()

    # Generate summary statistics
    print("   📈 Calculating comprehensive statistics...")
    summary = generate_summary_stats()

    # Add footer
    footer = generate_footer()

    # Combine everything
    full_report = f"{detailed_table}\n\n{summary}\n\n{footer}"

    # Save to file
    filename = "model_comparison_table.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(full_report)

    print(f"✅ Model comparison table saved to {filename}")
    print(f"📊 Total models documented: {len(COMPLETE_MODEL_CAPABILITIES)}")

    # Show key stats
    provider_counts = {}
    for caps in COMPLETE_MODEL_CAPABILITIES.values():
        provider = caps.provider
        provider_counts[provider] = provider_counts.get(provider, 0) + 1

    print("\n📈 Quick Summary:")
    for provider, count in sorted(provider_counts.items()):
        print(f"   {provider.capitalize()}: {count} models")

    # Show capability highlights
    tool_models = sum(
        1 for c in COMPLETE_MODEL_CAPABILITIES.values() if c.supports_tools
    )
    vision_models = sum(
        1 for c in COMPLETE_MODEL_CAPABILITIES.values() if c.supports_vision
    )
    reasoning_models = sum(
        1 for c in COMPLETE_MODEL_CAPABILITIES.values() if c.supports_reasoning
    )

    print("\n🎯 Capability Highlights:")
    print(f"   🛠️  Tool-capable: {tool_models} models")
    print(f"   👁️  Vision-capable: {vision_models} models")
    print(f"   🧠 Reasoning-capable: {reasoning_models} models")

    print(f"\n📖 Open {filename} to see the complete analysis!")


if __name__ == "__main__":
    main()
