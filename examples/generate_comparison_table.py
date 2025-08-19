#!/usr/bin/env python3
"""Generate a comprehensive model comparison table in Markdown format."""

from alea_llm_client.llms import (
    COMPLETE_MODEL_CAPABILITIES,
    generate_capability_table,
)


def generate_detailed_table():
    """Generate a detailed comparison table with all key capabilities."""
    
    lines = []
    lines.append("# Complete Model Capability Matrix")
    lines.append("")
    lines.append("Generated from alea-llm-client model registry")
    lines.append("")
    
    # Create the header
    lines.append("| Provider | Model | Tier | Status | Context | Max Output | Tools | Vision | Audio | Reasoning | Thinking | Grammar | Search |")
    lines.append("|----------|-------|------|--------|---------|------------|-------|--------|-------|-----------|----------|---------|--------|")
    
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
                    return f"{num//1000000}M"
                elif num >= 1000:
                    return f"{num//1000}K"
                else:
                    return str(num)
            
            context_str = format_tokens(caps.context_window)
            output_str = format_tokens(output)
            
            # Build the row
            row = [
                caps.provider,
                name,
                caps.tier.value,
                caps.status.value,
                context_str,
                output_str,
                "✓" if caps.supports_tools else "",
                "✓" if caps.supports_vision else "",
                "✓" if caps.supports_audio_input or caps.supports_audio_output else "",
                "✓" if caps.supports_reasoning else "",
                "✓" if caps.supports_thinking else "",
                "✓" if caps.supports_grammar else "",
                "✓" if caps.supports_search else "",
            ]
            
            lines.append("| " + " | ".join(row) + " |")
    
    lines.append("")
    lines.append("## Legend")
    lines.append("")
    lines.append("- **Tier**: nano < mini < small/flash < medium/standard < large/pro < heavy")
    lines.append("- **Status**: ga (Generally Available), preview, beta, experimental, legacy, deprecated")
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
    for name, caps in COMPLETE_MODEL_CAPABILITIES.items():
        provider = caps.provider
        by_provider[provider] = by_provider.get(provider, 0) + 1
    
    lines.append("### Models by Provider")
    for provider, count in sorted(by_provider.items()):
        lines.append(f"- {provider}: {count} models")
    
    lines.append(f"- **Total**: {len(COMPLETE_MODEL_CAPABILITIES)} models")
    lines.append("")
    
    # Count capabilities
    capability_counts = {
        "Tool/Function calling": sum(1 for c in COMPLETE_MODEL_CAPABILITIES.values() if c.supports_tools),
        "Structured output (JSON)": sum(1 for c in COMPLETE_MODEL_CAPABILITIES.values() if c.supports_structured_output),
        "Vision/Image input": sum(1 for c in COMPLETE_MODEL_CAPABILITIES.values() if c.supports_vision),
        "Audio capabilities": sum(1 for c in COMPLETE_MODEL_CAPABILITIES.values() if c.supports_audio_input or c.supports_audio_output),
        "Reasoning (O-series)": sum(1 for c in COMPLETE_MODEL_CAPABILITIES.values() if c.supports_reasoning),
        "Thinking mode": sum(1 for c in COMPLETE_MODEL_CAPABILITIES.values() if c.supports_thinking),
        "Grammar constraints": sum(1 for c in COMPLETE_MODEL_CAPABILITIES.values() if c.supports_grammar),
        "Web search": sum(1 for c in COMPLETE_MODEL_CAPABILITIES.values() if c.supports_search),
        "Fine-tuning": sum(1 for c in COMPLETE_MODEL_CAPABILITIES.values() if c.supports_fine_tuning),
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
    lines.append(f"- 50K-200K tokens: {sum(1 for c in contexts if 50000 <= c < 200000)} models")
    lines.append(f"- 200K-1M tokens: {sum(1 for c in contexts if 200000 <= c < 1000000)} models")
    lines.append(f"- ≥ 1M tokens: {sum(1 for c in contexts if c >= 1000000)} models")
    
    return "\n".join(lines)


def main():
    """Generate and save the comparison tables."""
    
    # Generate the detailed table
    detailed_table = generate_detailed_table()
    
    # Generate summary statistics
    summary = generate_summary_stats()
    
    # Combine everything
    full_report = f"{detailed_table}\n\n{summary}"
    
    # Save to file
    with open("model_comparison_table.md", "w") as f:
        f.write(full_report)
    
    print("Model comparison table saved to model_comparison_table.md")
    print(f"Total models documented: {len(COMPLETE_MODEL_CAPABILITIES)}")
    
    # Also print summary to console
    print("\n" + summary)


if __name__ == "__main__":
    main()