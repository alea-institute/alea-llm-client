# Implementation Plan: Latest Models and API Features

## Overview

This plan addresses the critical gaps identified in `API_GAPS_ANALYSIS.md` by implementing support for GPT-5, Claude 4, and their associated API features in our `httpx`-based ALEA LLM Client.

## Phase 1: URGENT FIXES (Sprint 1)

### 1.1 Model Support Updates

#### OpenAI Models
**File**: `alea_llm_client/llms/models/openai_model.py`
- Update default model from `gpt-4o` to `gpt-5-chat-latest`
- Add model validation constants

#### Anthropic Models  
**File**: `alea_llm_client/llms/models/anthropic_model.py`
- Update default model from `claude-3-5-sonnet-20240620` to `claude-sonnet-4-20250514`
- Add model validation constants

### 1.2 Critical Parameter Support

#### OpenAI: Reasoning Effort (CRITICAL)
**Target**: o-series models (o1, o3, o4) require this parameter
**Values**: `"minimal"` (fastest), `"low"`, `"medium"`, `"high"` (thorough chain-of-thought)
**Implementation**:
```python
# Add to openai_compatible_model.py _make_request()
if "reasoning_effort" in kwargs:
    effort = kwargs.pop("reasoning_effort")
    if effort in ["minimal", "low", "medium", "high"]:
        request_body["reasoning_effort"] = effort
    else:
        raise ValueError(f"Invalid reasoning_effort: {effort}")
```

#### OpenAI: Verbosity Control (HIGH PRIORITY)
**Target**: Controls response length while maintaining completeness
**Values**: `"low"` (concise), `"medium"`, `"high"` (comprehensive)
**Implementation**:
```python
# Add to openai_compatible_model.py _make_request()
if "verbosity" in kwargs:
    verbosity = kwargs.pop("verbosity")
    if verbosity in ["low", "medium", "high"]:
        request_body["verbosity"] = verbosity
    else:
        raise ValueError(f"Invalid verbosity: {verbosity}")
```

#### OpenAI: max_completion_tokens (BREAKING CHANGE)
**Target**: Replace deprecated `max_tokens` for o-series models
**Implementation**:
```python
# Add parameter translation in openai_compatible_model.py
if "max_completion_tokens" in kwargs:
    request_body["max_completion_tokens"] = kwargs.pop("max_completion_tokens")
elif "max_tokens" in kwargs:
    # Warn about deprecation
    self.logger.warning("max_tokens is deprecated, use max_completion_tokens")
    request_body["max_tokens"] = kwargs.pop("max_tokens")
```

#### Anthropic: Thinking Configuration (CRITICAL)
**Target**: Claude 4 extended thinking capabilities
**Implementation**:
```python
# Add thinking parameter to anthropic_model.py _make_request()
if "thinking" in kwargs:
    thinking_config = kwargs.pop("thinking")
    if isinstance(thinking_config, dict):
        request_body["thinking"] = thinking_config
    elif thinking_config is True:
        request_body["thinking"] = {"enabled": True}
    elif thinking_config is False:
        request_body["thinking"] = {"enabled": False}
```

### 1.3 Response Handling Updates

#### Thinking Blocks (Anthropic)
**File**: `alea_llm_client/llms/models/anthropic_model.py`
**Implementation**:
```python
@staticmethod
def _get_chat_choices(response_data: dict) -> List[str]:
    """Enhanced to handle thinking blocks."""
    content = response_data.get("content", [])
    if not content:
        return [""]
    
    # Extract text blocks, skip thinking blocks for now
    text_blocks = [
        block.get("text", "") 
        for block in content 
        if block.get("type") == "text"
    ]
    return text_blocks if text_blocks else [""]

def _get_thinking_content(self, response_data: dict) -> Optional[str]:
    """Extract thinking content from response."""
    content = response_data.get("content", [])
    thinking_blocks = [
        block.get("text", "")
        for block in content
        if block.get("type") == "thinking"
    ]
    return "\n".join(thinking_blocks) if thinking_blocks else None
```

## Phase 2: HIGH PRIORITY (Sprint 2)

### 2.1 Performance Optimizations

#### OpenAI Prompt Caching
**Implementation**:
```python
# Add to openai_compatible_model.py
if "prompt_cache_key" in kwargs:
    headers["Prompt-Cache-Key"] = kwargs.pop("prompt_cache_key")
```

#### Parallel Tool Calling
**Implementation**:
```python
# Add to openai_compatible_model.py _make_request()
if "parallel_tool_calls" in kwargs:
    request_body["parallel_tool_calls"] = kwargs.pop("parallel_tool_calls")
```

### 2.2 Enhanced Response Formats

#### Structured JSON Schema (OpenAI)
**Implementation**:
```python
# Enhance _json() method in openai_model.py
def _json(self, *args, **kwargs):
    # Support both legacy and new formats
    if "response_format" not in kwargs:
        kwargs["response_format"] = {"type": "json_object"}
    elif isinstance(kwargs["response_format"], dict):
        # Allow json_schema format
        if kwargs["response_format"].get("type") == "json_schema":
            # Keep as-is for structured output
            pass
    return super()._json(*args, **kwargs)
```

### 2.3 Model Response Enhancements

#### Extended ModelResponse Class
**File**: `alea_llm_client/llms/models/base_ai_model.py`
**Implementation**:
```python
@dataclass
class ModelResponse:
    choices: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    text: Optional[str] = None
    # NEW FIELDS
    thinking: Optional[str] = None  # Anthropic thinking content
    audio_url: Optional[str] = None  # OpenAI audio content
    reasoning_tokens: Optional[int] = None  # o-series reasoning tokens
```

## Phase 3: MEDIUM PRIORITY (Sprint 3)

### 3.1 Audio Modality Support (OpenAI)

#### Modalities Parameter
**Implementation**:
```python
# Add to openai_compatible_model.py
if "modalities" in kwargs:
    modalities = kwargs.pop("modalities")
    if isinstance(modalities, list):
        request_body["modalities"] = modalities
```

#### Audio Response Handling
**Implementation**:
```python
def _handle_audio_response(self, response_data: dict) -> Optional[str]:
    """Extract audio content from response."""
    choices = response_data.get("choices", [])
    for choice in choices:
        message = choice.get("message", {})
        if message.get("audio"):
            return message["audio"].get("url")
    return None
```

### 3.2 Web Search Integration (OpenAI)

#### Web Search Options
**Implementation**:
```python
# Add to openai_compatible_model.py
if "web_search_options" in kwargs:
    web_search = kwargs.pop("web_search_options")
    request_body["web_search_options"] = web_search
```

### 3.3 Prediction Optimization (OpenAI)

#### Prediction Parameter
**Implementation**:
```python
# Add to openai_compatible_model.py
if "prediction" in kwargs:
    prediction = kwargs.pop("prediction")
    request_body["prediction"] = prediction
```

## Implementation Strategy

### Development Approach
1. **Backward Compatibility**: All changes maintain existing API
2. **Gradual Migration**: Add new parameters without removing old ones
3. **Comprehensive Testing**: Test every new feature with real API calls
4. **Documentation Updates**: Update all docstrings and examples

### Code Organization

#### New Constants Files
**File**: `alea_llm_client/llms/constants/openai_models.py`
```python
# OpenAI Model Constants
GPT_5_MODELS = [
    "gpt-5", "gpt-5-mini", "gpt-5-nano",
    "gpt-5-2025-08-07", "gpt-5-mini-2025-08-07", "gpt-5-nano-2025-08-07",
    "gpt-5-chat-latest"
]

GPT_4_1_MODELS = [
    "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
    "gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14", "gpt-4.1-nano-2025-04-14"
]

O_SERIES_MODELS = [
    "o1", "o1-2024-12-17", "o1-preview", "o1-preview-2024-09-12",
    "o1-mini", "o1-mini-2024-09-12",
    "o3", "o3-2025-04-16", "o3-mini", "o3-mini-2025-01-31",
    "o4-mini", "o4-mini-2025-04-16"
]

REASONING_EFFORT_VALUES = ["minimal", "low", "medium", "high"]
VERBOSITY_VALUES = ["low", "medium", "high"]

# Parameter combinations for optimization
SPEED_OPTIMIZED = {"reasoning_effort": "minimal", "verbosity": "low"}
BALANCED = {"reasoning_effort": "medium", "verbosity": "medium"}  
THOROUGH = {"reasoning_effort": "high", "verbosity": "high"}
```

**File**: `alea_llm_client/llms/constants/anthropic_models.py`
```python
# Anthropic Model Constants
CLAUDE_4_MODELS = [
    "claude-sonnet-4-20250514", "claude-sonnet-4-0", "claude-4-sonnet-20250514",
    "claude-opus-4-0", "claude-opus-4-20250514", "claude-4-opus-20250514",
    "claude-opus-4-1-20250805"
]

CLAUDE_3_7_MODELS = [
    "claude-3-7-sonnet-latest", "claude-3-7-sonnet-20250219"
]

THINKING_ENABLED_MODELS = CLAUDE_4_MODELS + CLAUDE_3_7_MODELS
```

### Testing Strategy

#### Real API Integration Tests
**File**: `tests/test_new_models.py`
```python
import pytest
from alea_llm_client.llms.models import OpenAIModel, AnthropicModel

def test_gpt5_basic_chat():
    """Test GPT-5 basic functionality."""
    model = OpenAIModel(model="gpt-5")
    response = model.chat("Hello, test GPT-5 capabilities")
    assert response.text is not None
    assert len(response.text) > 0

def test_claude4_thinking():
    """Test Claude 4 thinking capabilities."""
    model = AnthropicModel(model="claude-sonnet-4-20250514")
    response = model.chat(
        "Solve this step by step: What is 127 * 83?",
        thinking={"enabled": True}
    )
    assert response.text is not None
    assert response.thinking is not None  # Should have thinking content

def test_o_series_reasoning_effort():
    """Test o-series models with reasoning effort."""
    model = OpenAIModel(model="o3-mini")
    response = model.chat(
        "Prove that the square root of 2 is irrational",
        reasoning_effort="high",
        max_completion_tokens=1000
    )
    assert response.text is not None
    assert "reasoning_tokens" in response.metadata

def test_verbosity_control():
    """Test verbosity parameter for response length control."""
    model = OpenAIModel(model="gpt-5")
    
    # Test low verbosity (concise)
    low_response = model.chat(
        "What is Raft consensus?",
        verbosity="low"
    )
    
    # Test high verbosity (comprehensive)  
    high_response = model.chat(
        "What is Raft consensus?",
        verbosity="high"
    )
    
    # High verbosity should produce longer responses
    assert len(high_response.text) > len(low_response.text)
    assert low_response.text is not None
    assert high_response.text is not None

def test_optimization_combinations():
    """Test parameter combinations for different use cases."""
    model = OpenAIModel(model="gpt-5")
    
    # Speed optimized
    fast_response = model.chat(
        "Optimize: SELECT * FROM orders WHERE created_at > '2024-01-01'",
        reasoning_effort="minimal",
        verbosity="low"
    )
    
    # Thorough analysis
    thorough_response = model.chat(
        "Optimize: SELECT * FROM orders WHERE created_at > '2024-01-01'",
        reasoning_effort="high", 
        verbosity="high"
    )
    
    assert fast_response.text is not None
    assert thorough_response.text is not None
    # Thorough should be longer and more detailed
    assert len(thorough_response.text) > len(fast_response.text)
```

### Deployment Checklist

#### Pre-deployment Validation
- [ ] All new models tested with real API calls
- [ ] Backward compatibility verified
- [ ] Type checking passes (`uvx ty check`)
- [ ] All tests pass (`uv run pytest`)
- [ ] Documentation updated
- [ ] CLAUDE.md updated with new capabilities

#### Breaking Changes Communication
1. **Model Defaults**: Announce change from GPT-4o to GPT-5
2. **Parameter Deprecation**: Warn about `max_tokens` → `max_completion_tokens`
3. **Response Format**: Document new response fields (thinking, audio_url, etc.)

## Success Metrics

### Functional Metrics
- [ ] 100% of GPT-5 models supported
- [ ] 100% of Claude 4 models supported  
- [ ] All critical parameters implemented (reasoning_effort, thinking)
- [ ] All tests passing with real API calls

### Performance Metrics
- [ ] Prompt caching reducing API costs by 20%+
- [ ] Reasoning effort controls working for o-series models
- [ ] Thinking configuration enabling Claude 4 advanced features

### Quality Metrics
- [ ] Zero breaking changes for existing users
- [ ] Complete type safety maintained
- [ ] Full test coverage for new features
- [ ] Comprehensive documentation

---

**Timeline**: 3 sprints (6 weeks) to complete all phases and achieve feature parity with latest OpenAI and Anthropic SDKs.