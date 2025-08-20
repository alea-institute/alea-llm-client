# SPDX-License-Identifier: (MIT)
__version__ = "0.2.3"
__author__ = "ALEA Institute (https://aleainstitute.ai)"
__license__ = "MIT"
__copyright__ = "Copyright 2024-2025, ALEA Institute"

# Core models
# Error handling
from .core import (
    ALEAAuthenticationError,
    ALEAError,
    ALEAModelError,
    ALEARetryExhaustedError,
)

# Logging utilities
from .core.logging import (
    DEFAULT_LOG_DIR,
    DEFAULT_LOG_FILE,
    DEFAULT_LOGGER,
    LoggerMixin,
    setup_logger,
)
from .llms import (
    AnthropicModel,
    BaseAIModel,
    GoogleModel,
    GrokModel,
    JSONModelResponse,
    ModelResponse,
    OpenAICompatibleModel,
    OpenAIModel,
    ResponseType,
    VLLMModel,
)

# Prompting utilities (public API)
from .llms.prompts.sections import (
    format_instructions,
    format_prompt,
)

# JSON utilities
from .llms.utils.json import (
    normalize_json_response,
    replace_jsons_refs_with_enum,
)

__all__ = [
    # Logging
    "DEFAULT_LOGGER",
    "DEFAULT_LOG_DIR",
    "DEFAULT_LOG_FILE",
    "ALEAAuthenticationError",
    "ALEAError",
    # Error handling
    "ALEAModelError",
    "ALEARetryExhaustedError",
    "AnthropicModel",
    # Models
    "BaseAIModel",
    "GoogleModel",
    "GrokModel",
    "JSONModelResponse",
    "LoggerMixin",
    "ModelResponse",
    "OpenAICompatibleModel",
    "OpenAIModel",
    "ResponseType",
    "VLLMModel",
    "format_instructions",
    # Prompt formatting
    "format_prompt",
    # JSON utilities
    "normalize_json_response",
    "replace_jsons_refs_with_enum",
    "setup_logger",
]
