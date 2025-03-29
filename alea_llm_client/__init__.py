# SPDX-License-Identifier: (MIT)
__version__ = "0.1.3"
__author__ = "ALEA Institute (https://aleainstitute.ai)"
__license__ = "MIT"
__copyright__ = "Copyright 2024-2025, ALEA Institute"

# Core models
from .llms import (
    BaseAIModel,
    OpenAICompatibleModel,
    VLLMModel,
    GrokModel,
    OpenAIModel,
    AnthropicModel,
    GoogleModel,
    ResponseType,
    ModelResponse,
    JSONModelResponse,
)

# Error handling
from .core import (
    ALEARetryExhaustedError,
    ALEAError,
    ALEAModelError,
    ALEAAuthenticationError,
)

# Logging utilities
from .core.logging import (
    DEFAULT_LOGGER,
    DEFAULT_LOG_DIR,
    DEFAULT_LOG_FILE,
    setup_logger,
    LoggerMixin,
)

# Prompting utilities
from .llms.prompts.formatters import (
    TokenType,
    format_prompt as format_model_prompt,
    format_prompt_llama3,
)
from .llms.prompts.sections import (
    format_prompt_sections,
    format_prompt,
    format_instructions,
    format_section_content,
)

# JSON utilities
from .llms.utils.json import (
    normalize_json_response,
    replace_jsons_refs_with_enum,
)

__all__ = [
    # Models
    "BaseAIModel",
    "OpenAICompatibleModel",
    "GrokModel",
    "VLLMModel",
    "OpenAIModel",
    "AnthropicModel",
    "GoogleModel",
    "ResponseType",
    "ModelResponse",
    "JSONModelResponse",
    # Error handling
    "ALEAModelError",
    "ALEAError",
    "ALEARetryExhaustedError",
    "ALEAAuthenticationError",
    # Logging
    "DEFAULT_LOGGER",
    "DEFAULT_LOG_DIR",
    "DEFAULT_LOG_FILE",
    "setup_logger",
    "LoggerMixin",
    # Prompt formatting
    "TokenType",
    "format_model_prompt",
    "format_prompt_llama3",
    "format_prompt_sections",
    "format_prompt",
    "format_instructions",
    "format_section_content",
    # JSON utilities
    "normalize_json_response",
    "replace_jsons_refs_with_enum",
]
