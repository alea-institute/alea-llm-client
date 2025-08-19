"""Comprehensive model capabilities matrix for all supported LLM providers.

This module contains detailed capability specifications for all models,
with properly typed fields and query functionality.

Sources:
- OpenAI documentation and API (2025)
- Anthropic documentation (2025)
- Google Gemini documentation (2025)
- xAI Grok documentation (2025)

Last Updated: August 2025
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class ModelStatus(str, Enum):
    """Model availability status."""

    GA = "ga"  # Generally Available
    PREVIEW = "preview"  # Preview/Beta
    BETA = "beta"  # Beta
    DEPRECATED = "deprecated"  # Deprecated
    LEGACY = "legacy"  # Legacy but still supported
    EXPERIMENTAL = "experimental"  # Experimental


class ModelTier(str, Enum):
    """Model size/performance tier."""

    NANO = "nano"  # Smallest
    MINI = "mini"  # Small
    SMALL = "small"  # Small (haiku)
    MEDIUM = "medium"  # Medium (sonnet)
    STANDARD = "standard"  # Standard size
    LARGE = "large"  # Large (opus)
    PRO = "pro"  # Professional/Advanced
    FLASH = "flash"  # Fast/Efficient (Google)
    HEAVY = "heavy"  # Most powerful


@dataclass
class DetailedCapabilities:
    """Detailed model capabilities with properly typed fields."""

    # Basic metadata
    provider: str
    model_name: str
    family: str
    tier: ModelTier
    status: ModelStatus = ModelStatus.GA
    release_date: Optional[str] = None  # ISO date
    knowledge_cutoff: Optional[str] = None  # ISO date

    # Token limits (numeric)
    context_window: int = 4096  # Max input tokens
    max_output_tokens: int = 4096  # Max output tokens
    default_output_tokens: int = 1024  # Default if not specified
    max_completion_tokens: Optional[int] = None  # For O-series models

    # Modality support (lists)
    input_modalities: List[str] = field(default_factory=lambda: ["text"])
    output_modalities: List[str] = field(default_factory=lambda: ["text"])

    # Core capabilities (boolean)
    supports_tools: bool = False  # Function calling/tools
    supports_structured_output: bool = False  # JSON mode/schema
    supports_streaming: bool = True  # SSE streaming
    supports_vision: bool = False  # Image input
    supports_audio_input: bool = False  # Audio input
    supports_audio_output: bool = False  # Audio generation
    supports_fine_tuning: bool = False  # Can be fine-tuned
    supports_logprobs: bool = False  # Log probabilities
    supports_search: bool = False  # Web search
    supports_batch: bool = True  # Batch API

    # Advanced features (boolean)
    supports_reasoning: bool = False  # O-series style reasoning
    supports_thinking: bool = False  # Claude/Gemini style thinking
    supports_grammar: bool = False  # Grammar constraints (GPT-5)
    supports_computer_use: bool = False  # Computer interaction
    supports_realtime: bool = False  # WebSocket/realtime

    # Reasoning/thinking parameters
    supports_reasoning_effort: bool = False  # Configurable reasoning
    thinking_budget_min: Optional[int] = None  # Min thinking tokens
    thinking_budget_max: Optional[int] = None  # Max thinking tokens

    # Output formats (list)
    output_formats: List[str] = field(default_factory=lambda: ["text"])

    # Special requirements
    requires_max_completion_tokens: bool = False  # O-series requirement
    requires_thinking_budget: bool = False  # Claude 4 requirement

    # Pricing tier (for reference)
    pricing_tier: Optional[str] = None  # low, medium, high, premium

    # Additional notes
    notes: Optional[str] = None


# Complete capability matrix for ALL models
COMPLETE_MODEL_CAPABILITIES: Dict[str, DetailedCapabilities] = {
    # ========== OPENAI MODELS ==========
    # GPT-5 Series (Latest flagship with grammar support)
    "gpt-5": DetailedCapabilities(
        provider="openai",
        model_name="gpt-5",
        family="gpt-5",
        tier=ModelTier.STANDARD,
        status=ModelStatus.GA,
        release_date="2025-08-07",
        knowledge_cutoff="2024-10",
        context_window=128000,
        max_output_tokens=4096,
        default_output_tokens=1024,
        supports_tools=True,
        supports_structured_output=True,
        supports_vision=True,
        supports_grammar=True,
        supports_streaming=True,
        supports_logprobs=True,
        output_formats=["text", "json", "json_schema"],
        pricing_tier="premium",
    ),
    "gpt-5-mini": DetailedCapabilities(
        provider="openai",
        model_name="gpt-5-mini",
        family="gpt-5",
        tier=ModelTier.MINI,
        status=ModelStatus.GA,
        release_date="2025-08-07",
        knowledge_cutoff="2024-10",
        context_window=128000,
        max_output_tokens=4096,
        default_output_tokens=1024,
        supports_tools=True,
        supports_structured_output=True,
        supports_vision=True,
        supports_grammar=True,
        supports_streaming=True,
        supports_logprobs=True,
        output_formats=["text", "json", "json_schema"],
        pricing_tier="medium",
    ),
    "gpt-5-nano": DetailedCapabilities(
        provider="openai",
        model_name="gpt-5-nano",
        family="gpt-5",
        tier=ModelTier.NANO,
        status=ModelStatus.GA,
        release_date="2025-08-07",
        knowledge_cutoff="2024-10",
        context_window=32000,
        max_output_tokens=2048,
        default_output_tokens=512,
        supports_tools=True,
        supports_structured_output=True,
        supports_grammar=True,
        supports_streaming=True,
        output_formats=["text", "json", "json_schema"],
        pricing_tier="low",
    ),
    "gpt-5-chat-latest": DetailedCapabilities(
        provider="openai",
        model_name="gpt-5-chat-latest",
        family="gpt-5",
        tier=ModelTier.STANDARD,
        status=ModelStatus.GA,
        release_date="2025-08-07",
        knowledge_cutoff="2024-10",
        context_window=128000,
        max_output_tokens=4096,
        default_output_tokens=1024,
        supports_tools=True,
        supports_structured_output=True,
        supports_vision=True,
        supports_grammar=True,
        supports_streaming=True,
        supports_logprobs=True,
        output_formats=["text", "json", "json_schema"],
        notes="Default ChatGPT model with dynamic routing",
    ),
    # GPT-4.1 Series (1M context)
    "gpt-4.1": DetailedCapabilities(
        provider="openai",
        model_name="gpt-4.1",
        family="gpt-4.1",
        tier=ModelTier.STANDARD,
        status=ModelStatus.GA,
        release_date="2025-04-14",
        knowledge_cutoff="2024-06",
        context_window=1000000,
        max_output_tokens=4096,
        default_output_tokens=1024,
        supports_tools=True,
        supports_structured_output=True,
        supports_vision=True,
        supports_streaming=True,
        supports_logprobs=True,
        output_formats=["text", "json", "json_schema"],
        notes="1M context but tool definitions >300k tokens may fail",
    ),
    "gpt-4.1-mini": DetailedCapabilities(
        provider="openai",
        model_name="gpt-4.1-mini",
        family="gpt-4.1",
        tier=ModelTier.MINI,
        status=ModelStatus.GA,
        release_date="2025-04-14",
        knowledge_cutoff="2024-06",
        context_window=1000000,
        max_output_tokens=4096,
        default_output_tokens=1024,
        supports_tools=True,
        supports_structured_output=True,
        supports_vision=True,
        supports_streaming=True,
        supports_logprobs=True,
        output_formats=["text", "json", "json_schema"],
    ),
    "gpt-4.1-nano": DetailedCapabilities(
        provider="openai",
        model_name="gpt-4.1-nano",
        family="gpt-4.1",
        tier=ModelTier.NANO,
        status=ModelStatus.GA,
        release_date="2025-04-14",
        knowledge_cutoff="2024-06",
        context_window=1000000,
        max_output_tokens=2048,
        default_output_tokens=512,
        supports_tools=True,
        supports_structured_output=True,
        supports_streaming=True,
        output_formats=["text", "json", "json_schema"],
    ),
    # O-Series Reasoning Models
    "o4-mini": DetailedCapabilities(
        provider="openai",
        model_name="o4-mini",
        family="o-series",
        tier=ModelTier.MINI,
        status=ModelStatus.GA,
        release_date="2025-04-16",
        context_window=128000,
        max_completion_tokens=65536,
        supports_reasoning=True,
        supports_reasoning_effort=True,
        requires_max_completion_tokens=True,
        supports_streaming=True,
        output_formats=["text"],
        notes="Reasoning model with configurable effort",
    ),
    "o3": DetailedCapabilities(
        provider="openai",
        model_name="o3",
        family="o-series",
        tier=ModelTier.STANDARD,
        status=ModelStatus.GA,
        release_date="2025-04-16",
        context_window=128000,
        max_completion_tokens=100000,
        supports_reasoning=True,
        supports_reasoning_effort=True,
        requires_max_completion_tokens=True,
        supports_streaming=True,
        output_formats=["text"],
    ),
    "o3-mini": DetailedCapabilities(
        provider="openai",
        model_name="o3-mini",
        family="o-series",
        tier=ModelTier.MINI,
        status=ModelStatus.GA,
        release_date="2025-01-31",
        context_window=200000,
        max_completion_tokens=100000,
        supports_reasoning=True,
        supports_reasoning_effort=True,
        requires_max_completion_tokens=True,
        supports_streaming=True,
        output_formats=["text"],
    ),
    "o3-pro": DetailedCapabilities(
        provider="openai",
        model_name="o3-pro",
        family="o-series",
        tier=ModelTier.PRO,
        status=ModelStatus.GA,
        release_date="2025-06-10",
        context_window=200000,
        max_completion_tokens=100000,
        supports_reasoning=True,
        supports_reasoning_effort=True,
        requires_max_completion_tokens=True,
        supports_streaming=True,
        output_formats=["text"],
        pricing_tier="premium",
    ),
    "o1": DetailedCapabilities(
        provider="openai",
        model_name="o1",
        family="o-series",
        tier=ModelTier.STANDARD,
        status=ModelStatus.GA,
        release_date="2024-12-17",
        context_window=200000,
        max_completion_tokens=100000,
        supports_reasoning=True,
        supports_reasoning_effort=True,
        requires_max_completion_tokens=True,
        supports_streaming=True,
        output_formats=["text"],
    ),
    "o1-mini": DetailedCapabilities(
        provider="openai",
        model_name="o1-mini",
        family="o-series",
        tier=ModelTier.MINI,
        status=ModelStatus.GA,
        release_date="2024-09-12",
        context_window=128000,
        max_completion_tokens=65536,
        supports_reasoning=True,
        supports_reasoning_effort=True,
        requires_max_completion_tokens=True,
        supports_streaming=True,
        output_formats=["text"],
    ),
    "o1-preview": DetailedCapabilities(
        provider="openai",
        model_name="o1-preview",
        family="o-series",
        tier=ModelTier.STANDARD,
        status=ModelStatus.PREVIEW,
        release_date="2024-09-12",
        context_window=128000,
        max_completion_tokens=32768,
        supports_reasoning=True,
        supports_reasoning_effort=True,
        requires_max_completion_tokens=True,
        supports_streaming=True,
        output_formats=["text"],
    ),
    "o1-pro": DetailedCapabilities(
        provider="openai",
        model_name="o1-pro",
        family="o-series",
        tier=ModelTier.PRO,
        status=ModelStatus.GA,
        release_date="2025-03-19",
        context_window=200000,
        max_completion_tokens=100000,
        supports_reasoning=True,
        supports_reasoning_effort=True,
        requires_max_completion_tokens=True,
        supports_streaming=True,
        output_formats=["text"],
        pricing_tier="premium",
    ),
    # GPT-4o Series
    "gpt-4o": DetailedCapabilities(
        provider="openai",
        model_name="gpt-4o",
        family="gpt-4o",
        tier=ModelTier.STANDARD,
        status=ModelStatus.GA,
        release_date="2024-05-13",
        context_window=128000,
        max_output_tokens=4096,
        input_modalities=["text", "image"],
        output_modalities=["text"],
        supports_tools=True,
        supports_structured_output=True,
        supports_vision=True,
        supports_streaming=True,
        supports_logprobs=True,
        output_formats=["text", "json", "json_schema"],
    ),
    "gpt-4o-mini": DetailedCapabilities(
        provider="openai",
        model_name="gpt-4o-mini",
        family="gpt-4o",
        tier=ModelTier.MINI,
        status=ModelStatus.GA,
        release_date="2024-07-18",
        context_window=128000,
        max_output_tokens=16384,
        input_modalities=["text", "image"],
        output_modalities=["text"],
        supports_tools=True,
        supports_structured_output=True,
        supports_vision=True,
        supports_streaming=True,
        supports_fine_tuning=True,
        output_formats=["text", "json", "json_schema"],
        pricing_tier="low",
    ),
    # Audio-capable models
    "gpt-4o-audio-preview": DetailedCapabilities(
        provider="openai",
        model_name="gpt-4o-audio-preview",
        family="gpt-4o",
        tier=ModelTier.STANDARD,
        status=ModelStatus.PREVIEW,
        release_date="2024-10-01",
        context_window=128000,
        max_output_tokens=4096,
        input_modalities=["text", "image", "audio"],
        output_modalities=["text", "audio"],
        supports_tools=True,
        supports_vision=True,
        supports_audio_input=True,
        supports_audio_output=True,
        supports_streaming=True,
        output_formats=["text", "audio"],
    ),
    # Search-enabled models
    "gpt-4o-search-preview": DetailedCapabilities(
        provider="openai",
        model_name="gpt-4o-search-preview",
        family="gpt-4o",
        tier=ModelTier.STANDARD,
        status=ModelStatus.PREVIEW,
        release_date="2025-03-11",
        context_window=128000,
        max_output_tokens=4096,
        supports_tools=True,
        supports_structured_output=True,
        supports_vision=True,
        supports_search=True,
        supports_streaming=True,
        output_formats=["text", "json", "json_schema"],
        notes="Web search enabled",
    ),
    # Legacy GPT-4
    "gpt-4-turbo": DetailedCapabilities(
        provider="openai",
        model_name="gpt-4-turbo",
        family="gpt-4",
        tier=ModelTier.STANDARD,
        status=ModelStatus.LEGACY,
        context_window=128000,
        max_output_tokens=4096,
        supports_tools=True,
        supports_structured_output=True,
        supports_vision=True,
        supports_streaming=True,
        output_formats=["text", "json"],
    ),
    "gpt-4": DetailedCapabilities(
        provider="openai",
        model_name="gpt-4",
        family="gpt-4",
        tier=ModelTier.STANDARD,
        status=ModelStatus.LEGACY,
        context_window=8192,
        max_output_tokens=4096,
        supports_tools=True,
        supports_streaming=True,
        output_formats=["text"],
    ),
    "gpt-4-32k": DetailedCapabilities(
        provider="openai",
        model_name="gpt-4-32k",
        family="gpt-4",
        tier=ModelTier.STANDARD,
        status=ModelStatus.LEGACY,
        context_window=32768,
        max_output_tokens=4096,
        supports_tools=True,
        supports_streaming=True,
        output_formats=["text"],
    ),
    # Legacy GPT-3.5
    "gpt-3.5-turbo": DetailedCapabilities(
        provider="openai",
        model_name="gpt-3.5-turbo",
        family="gpt-3.5",
        tier=ModelTier.STANDARD,
        status=ModelStatus.LEGACY,
        context_window=16385,
        max_output_tokens=4096,
        supports_tools=True,
        supports_structured_output=True,
        supports_streaming=True,
        output_formats=["text", "json"],
    ),
    "gpt-3.5-turbo-16k": DetailedCapabilities(
        provider="openai",
        model_name="gpt-3.5-turbo-16k",
        family="gpt-3.5",
        tier=ModelTier.STANDARD,
        status=ModelStatus.LEGACY,
        context_window=16384,
        max_output_tokens=4096,
        supports_tools=True,
        supports_streaming=True,
        output_formats=["text"],
    ),
    # ========== ANTHROPIC MODELS ==========
    # Claude 4 Series (with thinking)
    "claude-sonnet-4-20250514": DetailedCapabilities(
        provider="anthropic",
        model_name="claude-sonnet-4-20250514",
        family="claude-4",
        tier=ModelTier.MEDIUM,
        status=ModelStatus.GA,
        release_date="2025-05-14",
        context_window=1000000,
        max_output_tokens=64000,
        default_output_tokens=4096,
        supports_tools=True,
        supports_structured_output=True,
        supports_vision=True,
        supports_thinking=True,
        thinking_budget_min=1024,
        thinking_budget_max=64000,
        supports_streaming=True,
        input_modalities=["text", "image"],
        output_formats=["text", "json"],
        notes="1M context in beta, 200k standard",
        pricing_tier="medium",
    ),
    "claude-opus-4-20250514": DetailedCapabilities(
        provider="anthropic",
        model_name="claude-opus-4-20250514",
        family="claude-4",
        tier=ModelTier.LARGE,
        status=ModelStatus.GA,
        release_date="2025-05-14",
        context_window=200000,
        max_output_tokens=32000,
        default_output_tokens=4096,
        supports_tools=True,
        supports_structured_output=True,
        supports_vision=True,
        supports_thinking=True,
        thinking_budget_min=1024,
        thinking_budget_max=32000,
        supports_streaming=True,
        input_modalities=["text", "image"],
        output_formats=["text", "json"],
        pricing_tier="premium",
    ),
    "claude-opus-4-1-20250805": DetailedCapabilities(
        provider="anthropic",
        model_name="claude-opus-4-1-20250805",
        family="claude-4",
        tier=ModelTier.LARGE,
        status=ModelStatus.GA,
        release_date="2025-08-05",
        context_window=200000,
        max_output_tokens=32000,
        default_output_tokens=4096,
        supports_tools=True,
        supports_structured_output=True,
        supports_vision=True,
        supports_thinking=True,
        thinking_budget_min=1024,
        thinking_budget_max=32000,
        supports_streaming=True,
        input_modalities=["text", "image"],
        output_formats=["text", "json"],
        pricing_tier="premium",
    ),
    # Claude 3.7 Series (with thinking)
    "claude-3-7-sonnet-latest": DetailedCapabilities(
        provider="anthropic",
        model_name="claude-3-7-sonnet-latest",
        family="claude-3.7",
        tier=ModelTier.MEDIUM,
        status=ModelStatus.GA,
        release_date="2025-02-19",
        context_window=200000,
        max_output_tokens=128000,
        default_output_tokens=4096,
        supports_tools=True,
        supports_structured_output=True,
        supports_vision=True,
        supports_thinking=True,
        thinking_budget_min=1024,
        thinking_budget_max=64000,
        supports_streaming=True,
        input_modalities=["text", "image"],
        output_formats=["text", "json"],
        notes="Extended output to 128k with beta header",
    ),
    "claude-3-7-sonnet-20250219": DetailedCapabilities(
        provider="anthropic",
        model_name="claude-3-7-sonnet-20250219",
        family="claude-3.7",
        tier=ModelTier.MEDIUM,
        status=ModelStatus.GA,
        release_date="2025-02-19",
        context_window=200000,
        max_output_tokens=128000,
        default_output_tokens=4096,
        supports_tools=True,
        supports_structured_output=True,
        supports_vision=True,
        supports_thinking=True,
        thinking_budget_min=1024,
        thinking_budget_max=64000,
        supports_streaming=True,
        input_modalities=["text", "image"],
        output_formats=["text", "json"],
    ),
    # Claude 3.5 Series
    "claude-3-5-sonnet-latest": DetailedCapabilities(
        provider="anthropic",
        model_name="claude-3-5-sonnet-latest",
        family="claude-3.5",
        tier=ModelTier.MEDIUM,
        status=ModelStatus.GA,
        context_window=200000,
        max_output_tokens=8192,
        default_output_tokens=4096,
        supports_tools=True,
        supports_structured_output=True,
        supports_vision=True,
        supports_streaming=True,
        input_modalities=["text", "image"],
        output_formats=["text", "json"],
    ),
    "claude-3-5-sonnet-20241022": DetailedCapabilities(
        provider="anthropic",
        model_name="claude-3-5-sonnet-20241022",
        family="claude-3.5",
        tier=ModelTier.MEDIUM,
        status=ModelStatus.GA,
        release_date="2024-10-22",
        context_window=200000,
        max_output_tokens=8192,
        default_output_tokens=4096,
        supports_tools=True,
        supports_structured_output=True,
        supports_vision=True,
        supports_streaming=True,
        input_modalities=["text", "image"],
        output_formats=["text", "json"],
    ),
    "claude-3-5-haiku-latest": DetailedCapabilities(
        provider="anthropic",
        model_name="claude-3-5-haiku-latest",
        family="claude-3.5",
        tier=ModelTier.SMALL,
        status=ModelStatus.GA,
        context_window=200000,
        max_output_tokens=8192,
        default_output_tokens=4096,
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        input_modalities=["text", "image"],
        output_formats=["text"],
        pricing_tier="low",
    ),
    "claude-3-5-haiku-20241022": DetailedCapabilities(
        provider="anthropic",
        model_name="claude-3-5-haiku-20241022",
        family="claude-3.5",
        tier=ModelTier.SMALL,
        status=ModelStatus.GA,
        release_date="2024-10-22",
        context_window=200000,
        max_output_tokens=8192,
        default_output_tokens=4096,
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        input_modalities=["text", "image"],
        output_formats=["text"],
        pricing_tier="low",
    ),
    # Claude 3 Series
    "claude-3-opus-latest": DetailedCapabilities(
        provider="anthropic",
        model_name="claude-3-opus-latest",
        family="claude-3",
        tier=ModelTier.LARGE,
        status=ModelStatus.LEGACY,
        context_window=200000,
        max_output_tokens=4096,
        default_output_tokens=4096,
        supports_tools=True,
        supports_structured_output=True,
        supports_vision=True,
        supports_streaming=True,
        input_modalities=["text", "image"],
        output_formats=["text", "json"],
        pricing_tier="high",
    ),
    "claude-3-opus-20240229": DetailedCapabilities(
        provider="anthropic",
        model_name="claude-3-opus-20240229",
        family="claude-3",
        tier=ModelTier.LARGE,
        status=ModelStatus.LEGACY,
        release_date="2024-02-29",
        context_window=200000,
        max_output_tokens=4096,
        default_output_tokens=4096,
        supports_tools=True,
        supports_structured_output=True,
        supports_vision=True,
        supports_streaming=True,
        input_modalities=["text", "image"],
        output_formats=["text", "json"],
    ),
    "claude-3-haiku-20240307": DetailedCapabilities(
        provider="anthropic",
        model_name="claude-3-haiku-20240307",
        family="claude-3",
        tier=ModelTier.SMALL,
        status=ModelStatus.LEGACY,
        release_date="2024-03-07",
        context_window=200000,
        max_output_tokens=4096,
        default_output_tokens=4096,
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        input_modalities=["text", "image"],
        output_formats=["text"],
        pricing_tier="low",
    ),
    # ========== GOOGLE MODELS ==========
    # Gemini 2.5 Series (with thinking)
    "gemini-2.5-flash": DetailedCapabilities(
        provider="google",
        model_name="gemini-2.5-flash",
        family="gemini-2.5",
        tier=ModelTier.FLASH,
        status=ModelStatus.GA,
        release_date="2025-03-01",
        context_window=1000000,
        max_output_tokens=8192,
        default_output_tokens=2048,
        input_modalities=["text", "image", "video", "audio"],
        output_modalities=["text"],
        supports_tools=True,
        supports_structured_output=True,
        supports_vision=True,
        supports_thinking=True,
        thinking_budget_min=4000,
        thinking_budget_max=32000,
        supports_streaming=True,
        supports_audio_input=True,
        output_formats=["text", "json"],
        notes="First Flash model with thinking",
        pricing_tier="low",
    ),
    "gemini-2.5-pro": DetailedCapabilities(
        provider="google",
        model_name="gemini-2.5-pro",
        family="gemini-2.5",
        tier=ModelTier.PRO,
        status=ModelStatus.GA,
        release_date="2025-03-01",
        context_window=2000000,
        max_output_tokens=8192,
        default_output_tokens=2048,
        input_modalities=["text", "image", "video", "audio"],
        output_modalities=["text"],
        supports_tools=True,
        supports_structured_output=True,
        supports_vision=True,
        supports_thinking=True,
        thinking_budget_min=4000,
        thinking_budget_max=64000,
        supports_streaming=True,
        supports_audio_input=True,
        output_formats=["text", "json"],
        notes="2M context coming soon",
        pricing_tier="high",
    ),
    # Gemini 2.0 Series
    "gemini-2.0-flash": DetailedCapabilities(
        provider="google",
        model_name="gemini-2.0-flash",
        family="gemini-2.0",
        tier=ModelTier.FLASH,
        status=ModelStatus.GA,
        context_window=1000000,
        max_output_tokens=8192,
        default_output_tokens=2048,
        input_modalities=["text", "image", "video", "audio"],
        output_modalities=["text"],
        supports_tools=True,
        supports_structured_output=True,
        supports_vision=True,
        supports_streaming=True,
        supports_audio_input=True,
        output_formats=["text", "json"],
        pricing_tier="low",
    ),
    "gemini-2.0-pro": DetailedCapabilities(
        provider="google",
        model_name="gemini-2.0-pro",
        family="gemini-2.0",
        tier=ModelTier.PRO,
        status=ModelStatus.GA,
        context_window=1000000,
        max_output_tokens=8192,
        default_output_tokens=2048,
        input_modalities=["text", "image", "video", "audio"],
        output_modalities=["text"],
        supports_tools=True,
        supports_structured_output=True,
        supports_vision=True,
        supports_streaming=True,
        supports_audio_input=True,
        output_formats=["text", "json"],
        pricing_tier="medium",
    ),
    "gemini-2.0-flash-exp": DetailedCapabilities(
        provider="google",
        model_name="gemini-2.0-flash-exp",
        family="gemini-2.0",
        tier=ModelTier.FLASH,
        status=ModelStatus.EXPERIMENTAL,
        context_window=1000000,
        max_output_tokens=8192,
        default_output_tokens=2048,
        input_modalities=["text", "image", "video", "audio"],
        output_modalities=["text"],
        supports_tools=True,
        supports_structured_output=True,
        supports_vision=True,
        supports_streaming=True,
        supports_audio_input=True,
        output_formats=["text", "json"],
        notes="Experimental features",
    ),
    # Gemini 1.5 Series (Previous gen)
    "gemini-1.5-flash": DetailedCapabilities(
        provider="google",
        model_name="gemini-1.5-flash",
        family="gemini-1.5",
        tier=ModelTier.FLASH,
        status=ModelStatus.LEGACY,
        context_window=1000000,
        max_output_tokens=8192,
        default_output_tokens=2048,
        input_modalities=["text", "image", "video", "audio"],
        output_modalities=["text"],
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        supports_audio_input=True,
        output_formats=["text", "json"],
    ),
    "gemini-1.5-pro": DetailedCapabilities(
        provider="google",
        model_name="gemini-1.5-pro",
        family="gemini-1.5",
        tier=ModelTier.PRO,
        status=ModelStatus.LEGACY,
        context_window=2000000,
        max_output_tokens=8192,
        default_output_tokens=2048,
        input_modalities=["text", "image", "video", "audio"],
        output_modalities=["text"],
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        supports_audio_input=True,
        output_formats=["text", "json"],
    ),
    # ========== GROK MODELS ==========
    "grok-4": DetailedCapabilities(
        provider="xai",
        model_name="grok-4",
        family="grok-4",
        tier=ModelTier.STANDARD,
        status=ModelStatus.GA,
        release_date="2025-07-10",
        knowledge_cutoff="2024-11",
        context_window=256000,
        max_output_tokens=8192,
        default_output_tokens=2048,
        supports_tools=True,
        supports_streaming=True,
        supports_reasoning=True,
        supports_search=True,
        output_formats=["text"],
        notes="Reasoning model with no non-reasoning mode",
        pricing_tier="high",
    ),
    "grok-4-heavy": DetailedCapabilities(
        provider="xai",
        model_name="grok-4-heavy",
        family="grok-4",
        tier=ModelTier.HEAVY,
        status=ModelStatus.GA,
        release_date="2025-07-10",
        knowledge_cutoff="2024-11",
        context_window=256000,
        max_output_tokens=8192,
        default_output_tokens=2048,
        supports_tools=True,
        supports_streaming=True,
        supports_reasoning=True,
        supports_search=True,
        output_formats=["text"],
        notes="Most powerful Grok model",
        pricing_tier="premium",
    ),
    "grok-3": DetailedCapabilities(
        provider="xai",
        model_name="grok-3",
        family="grok-3",
        tier=ModelTier.STANDARD,
        status=ModelStatus.GA,
        release_date="2025-02-01",
        context_window=131072,
        max_output_tokens=8192,
        default_output_tokens=2048,
        supports_tools=True,
        supports_streaming=True,
        supports_reasoning=True,
        supports_vision=True,
        input_modalities=["text", "image"],
        output_formats=["text"],
        notes="1M context advertised but 131k in API",
    ),
    "grok-3-mini": DetailedCapabilities(
        provider="xai",
        model_name="grok-3-mini",
        family="grok-3",
        tier=ModelTier.MINI,
        status=ModelStatus.GA,
        release_date="2025-02-01",
        context_window=131072,
        max_output_tokens=8192,
        default_output_tokens=2048,
        supports_tools=True,
        supports_streaming=True,
        output_formats=["text"],
        pricing_tier="low",
    ),
    "grok-2": DetailedCapabilities(
        provider="xai",
        model_name="grok-2",
        family="grok-2",
        tier=ModelTier.STANDARD,
        status=ModelStatus.GA,
        context_window=32768,
        max_output_tokens=4096,
        default_output_tokens=1024,
        supports_tools=True,
        supports_streaming=True,
        output_formats=["text"],
    ),
    "grok-2-mini": DetailedCapabilities(
        provider="xai",
        model_name="grok-2-mini",
        family="grok-2",
        tier=ModelTier.MINI,
        status=ModelStatus.GA,
        context_window=32768,
        max_output_tokens=4096,
        default_output_tokens=1024,
        supports_streaming=True,
        output_formats=["text"],
        pricing_tier="low",
    ),
    "grok-2-1212": DetailedCapabilities(
        provider="xai",
        model_name="grok-2-1212",
        family="grok-2",
        tier=ModelTier.STANDARD,
        status=ModelStatus.GA,
        release_date="2024-12-12",
        context_window=32768,
        max_output_tokens=4096,
        default_output_tokens=1024,
        supports_tools=True,
        supports_streaming=True,
        output_formats=["text"],
    ),
    "grok-beta": DetailedCapabilities(
        provider="xai",
        model_name="grok-beta",
        family="grok-beta",
        tier=ModelTier.STANDARD,
        status=ModelStatus.BETA,
        context_window=32768,
        max_output_tokens=4096,
        default_output_tokens=1024,
        supports_streaming=True,
        output_formats=["text"],
        notes="Beta features",
    ),
}


# Query Functions


def get_models_with_context_window_gte(min_tokens: int) -> List[str]:
    """Get all models with context window >= specified tokens."""
    return [
        name
        for name, caps in COMPLETE_MODEL_CAPABILITIES.items()
        if caps.context_window >= min_tokens
    ]


def get_models_with_capability(capability: str) -> List[str]:
    """Get all models that support a specific capability.

    Args:
        capability: One of: tools, structured_output, vision, audio, audio_input,
                   audio_output, reasoning, thinking, grammar, search, etc.
    """
    capability_map = {
        "tools": lambda c: c.supports_tools,
        "structured_output": lambda c: c.supports_structured_output,
        "vision": lambda c: c.supports_vision,
        "audio": lambda c: c.supports_audio_input or c.supports_audio_output,
        "audio_input": lambda c: c.supports_audio_input,
        "audio_output": lambda c: c.supports_audio_output,
        "reasoning": lambda c: c.supports_reasoning,
        "thinking": lambda c: c.supports_thinking,
        "grammar": lambda c: c.supports_grammar,
        "search": lambda c: c.supports_search,
        "streaming": lambda c: c.supports_streaming,
        "fine_tuning": lambda c: c.supports_fine_tuning,
        "realtime": lambda c: c.supports_realtime,
        "computer_use": lambda c: c.supports_computer_use,
    }

    check_fn = capability_map.get(capability)
    if not check_fn:
        raise ValueError(f"Unknown capability: {capability}")

    return [
        name for name, caps in COMPLETE_MODEL_CAPABILITIES.items() if check_fn(caps)
    ]


def get_models_by_tier(tier: ModelTier) -> List[str]:
    """Get all models of a specific tier."""
    return [
        name for name, caps in COMPLETE_MODEL_CAPABILITIES.items() if caps.tier == tier
    ]


def get_models_by_provider(provider: str) -> List[str]:
    """Get all models from a specific provider."""
    return [
        name
        for name, caps in COMPLETE_MODEL_CAPABILITIES.items()
        if caps.provider.lower() == provider.lower()
    ]


def get_model_details(model_name: str) -> Optional[DetailedCapabilities]:
    """Get detailed capabilities for a specific model."""
    return COMPLETE_MODEL_CAPABILITIES.get(model_name)


def compare_models(model_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """Compare capabilities of multiple models.

    Returns a comparison matrix with model names as keys.
    """
    comparison = {}

    for name in model_names:
        caps = COMPLETE_MODEL_CAPABILITIES.get(name)
        if caps:
            comparison[name] = {
                "provider": caps.provider,
                "tier": caps.tier.value,
                "context_window": caps.context_window,
                "max_output": caps.max_output_tokens,
                "tools": caps.supports_tools,
                "vision": caps.supports_vision,
                "audio": caps.supports_audio_input or caps.supports_audio_output,
                "reasoning": caps.supports_reasoning,
                "thinking": caps.supports_thinking,
                "status": caps.status.value,
            }

    return comparison


def filter_models(
    min_context: Optional[int] = None,
    max_context: Optional[int] = None,
    min_output: Optional[int] = None,
    providers: Optional[List[str]] = None,
    tiers: Optional[List[ModelTier]] = None,
    capabilities: Optional[List[str]] = None,
    exclude_deprecated: bool = True,
) -> List[str]:
    """Advanced filtering with multiple criteria.

    Args:
        min_context: Minimum context window size
        max_context: Maximum context window size
        min_output: Minimum output token limit
        providers: List of providers to include
        tiers: List of tiers to include
        capabilities: Required capabilities
        exclude_deprecated: Whether to exclude deprecated models

    Returns:
        List of model names matching all criteria
    """
    results = []

    for name, caps in COMPLETE_MODEL_CAPABILITIES.items():
        # Check context window
        if min_context and caps.context_window < min_context:
            continue
        if max_context and caps.context_window > max_context:
            continue

        # Check output tokens
        if min_output:
            output_tokens = caps.max_completion_tokens or caps.max_output_tokens
            if output_tokens < min_output:
                continue

        # Check provider
        if providers and caps.provider not in providers:
            continue

        # Check tier
        if tiers and caps.tier not in tiers:
            continue

        # Check status
        if exclude_deprecated and caps.status in [
            ModelStatus.DEPRECATED,
            ModelStatus.LEGACY,
        ]:
            continue

        # Check capabilities
        if capabilities:
            capability_checks = {
                "tools": caps.supports_tools,
                "structured_output": caps.supports_structured_output,
                "vision": caps.supports_vision,
                "audio": caps.supports_audio_input or caps.supports_audio_output,
                "reasoning": caps.supports_reasoning,
                "thinking": caps.supports_thinking,
                "grammar": caps.supports_grammar,
                "search": caps.supports_search,
            }

            if not all(capability_checks.get(cap, False) for cap in capabilities):
                continue

        results.append(name)

    return results


def generate_capability_table() -> str:
    """Generate a markdown table comparing all models."""
    lines = []
    lines.append(
        "| Model | Provider | Tier | Context | Output | Tools | Vision | Audio | Reasoning | Thinking | Status |"
    )
    lines.append(
        "|-------|----------|------|---------|--------|-------|--------|-------|-----------|----------|--------|"
    )

    for name, caps in sorted(COMPLETE_MODEL_CAPABILITIES.items()):
        output = caps.max_completion_tokens or caps.max_output_tokens
        lines.append(
            f"| {name} | {caps.provider} | {caps.tier.value} | "
            f"{caps.context_window:,} | {output:,} | "
            f"{'✓' if caps.supports_tools else '✗'} | "
            f"{'✓' if caps.supports_vision else '✗'} | "
            f"{'✓' if caps.supports_audio_input or caps.supports_audio_output else '✗'} | "
            f"{'✓' if caps.supports_reasoning else '✗'} | "
            f"{'✓' if caps.supports_thinking else '✗'} | "
            f"{caps.status.value} |"
        )

    return "\n".join(lines)
