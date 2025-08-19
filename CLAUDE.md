# ALEA LLM Client - Python 3.13+ Development Guide

**ALWAYS FOLLOW THE WORKFLOW OUTLINED BELOW**
- Never make things up or guess. Research data models, libraries, and APIs in the local virtual environment using the tools below.
- Search the web or ask for help if you cannot find sufficient information.
- Only write code if you understand the external dependencies or libraries you are using. Never make up classes, methods, enums, data models, etc.
- Wherever reasonable, follow TDD: begin by writing **REAL INTEGRATION TESTS** with `pytest` that are expected to fail. Create staged tests that allow you to progressively assess functionality.
- Use `uv run python -c ...` regularly to introspect objects or debug your work.
- If you mock or fake test data, you may cause real users to be emotionally or physically harmed. **YOU MUST NEVER MOCK OR FAKE DATA.**
- Always check your work using `uvx ruff` and `uvx ty` as outlined below.
- **NEVER TELL ME YOU ARE DONE IF YOU HAVE NOT CHECKED YOUR WORK.**

## THIS IS REAL - THIS IS PRODUCTION

This is a **REAL** project - it will be used in production by real people who will depend on it to work properly. If you fake or mock tests, you will harm users. If you create brittle or dangerous code, you will harm users. If you are lazy and fail to follow the instructions to lint or test your code, you will harm users. **NEVER FORGET THAT YOU ARE WORKING ON A REAL PROJECT AND YOUR CHOICES IMPACT REAL PEOPLE.**

## CRITICAL DEPENDENCY CONSTRAINTS

**⚠️ THIS PROJECT MUST ONLY HAVE TWO RUNTIME DEPENDENCIES: `httpx` AND `pydantic` ⚠️**

- **NEVER ADD ANY OTHER RUNTIME DEPENDENCIES** without explicit approval
- **NO REQUESTS** - Use `httpx` only
- **NO OPENAI SDK** - We implement our own OpenAI-compatible client
- **NO ANTHROPIC SDK** - We implement our own Anthropic client
- **NO GOOGLE SDK** - We implement our own Google client
- The entire point of this library is to be minimal and lightweight
- Check `pyproject.toml` dependencies section - it should only list `httpx` and `pydantic`

## API Keys Are Already Configured

**🔑 API KEYS ARE ALREADY SET UP IN `~/.alea/keys/` 🔑**

- **DO NOT ASK FOR API KEYS** - they are already configured
- Anthropic key: `~/.alea/keys/anthropic`
- OpenAI key: `~/.alea/keys/openai`  
- Google key: `~/.alea/keys/google`
- The models will automatically read from these locations
- **NEVER HARDCODE API KEYS IN CODE OR TESTS**

## Best Practices and Standards

When in doubt, always follow best practices and implement standards. You should write code like it will be audited or reviewed by an expert.

## Manage Python with `uv` and `uvx`

**ALWAYS** use `uv` and `uvx` to manage and run Python environments.

*   Install dependencies: `uv sync`
*   Add a development dependency: `uv add --group dev ...`
*   Run a python script: `uv run python ...`
*   Run a python script with PYTHONPATH: `PYTHONPATH=... uv run python ...`
*   Run an ephemeral tool: `uvx {tool} ...`
*   Lint and fix with `ruff`: `uvx ruff check --fix ...`
*   Lint with `ruff`: `uvx ruff check ...`
*   Format with `ruff`: `uvx ruff format ...`
*   Check types with `ty`: `uv run uvx ty check alea_llm_client/`

## Always Examine and Introspect

When you encounter an unknown library, class, or method, **ALWAYS EXAMINE AND INTROSPECT** before writing code.

### With Python

*   Inspect object attributes and methods: `uv run python -c "import httpx; print([x for x in dir(httpx) if not x.startswith('_')])"`
*   Get function signatures: `uv run python -c "import inspect, httpx; print(inspect.signature(httpx.Client.post))"`
*   Explore module structure: `uv run python -c "import pkgutil, httpx; [print(f'{name}: {ispkg}') for _, name, ispkg in pkgutil.iter_modules(httpx.__path__, 'httpx.')]"`
*   Find where a module is located: `uv run python -c "import importlib.util; print(importlib.util.find_spec('httpx').origin)"`

### With `grep`, `ripgrep`, `sed`, `awk`

*   Search the virtual environment with ripgrep: `rg -n -uu -tpy "class.*Exception" .venv/lib/`
*   Search the virtual environment with grep: `grep -niR "class.*Exception" .venv/lib/`
*   Find where a class is defined: `rg -uu -tpy "^class Response" .venv/lib/`
*   Find function and show implementation context: `grep -n -A 20 -B 10 "def process_data" *.py`

## Project Architecture

This is an LLM client abstraction library with the following structure:

```
alea_llm_client/
├── core/
│   ├── exceptions.py      # Custom exception classes
│   └── logging/           # Logging utilities
├── llms/
│   ├── models/            # Model implementations
│   │   ├── base_ai_model.py          # Abstract base class
│   │   ├── openai_compatible_model.py # OpenAI-compatible base
│   │   ├── anthropic_model.py         # Anthropic implementation
│   │   ├── openai_model.py           # OpenAI implementation
│   │   ├── google_model.py           # Google Vertex implementation
│   │   ├── grok_model.py             # Grok implementation
│   │   └── vllm_model.py             # VLLM implementation
│   ├── prompts/           # Prompt formatting utilities
│   └── utils/             # Utility functions
└── tests/                 # Test files
```

## Key Design Patterns

### 1. Model Inheritance Hierarchy
- `BaseAIModel` - Abstract base class defining the interface
- `OpenAICompatibleModel` - Base for OpenAI-compatible APIs (VLLM, OpenAI, etc.)
- Specific model classes inherit from these bases

### 2. Response Types
- `ModelResponse` - Plain text responses
- `JSONModelResponse` - JSON structured responses  
- `BaseModel` - Pydantic validated responses

### 3. Caching System
- Automatic response caching based on request parameters
- Cache paths use hashes of endpoint + model + parameters
- Configurable cache directory per model

## Testing

### Pytest Setup

**ALWAYS** use `pytest` for testing. The project is already configured.

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=alea_llm_client --cov-report=html

# Run specific test file
uv run pytest tests/test_anthropic.py
```

### Testing Real API Calls

**CRITICAL: Use real API calls, never mock external services**

```python
import pytest
from alea_llm_client.llms.models.anthropic_model import AnthropicModel

def test_anthropic_chat_completion():
    """Test real Anthropic API call."""
    model = AnthropicModel(model="claude-3-5-sonnet-20240620")
    
    response = model.chat("Hello, how are you?")
    
    assert response.text is not None
    assert len(response.text) > 0
    assert response.choices[0] is not None
    assert "usage" in response.metadata

@pytest.mark.asyncio
async def test_anthropic_async_chat():
    """Test async Anthropic API call."""
    model = AnthropicModel(model="claude-3-5-sonnet-20240620")
    
    response = await model.chat_async("What is Python?")
    
    assert response.text is not None
    assert len(response.text) > 0
```

## Quality Assurance Checklist

**BEFORE SUBMITTING ANY CODE, RUN ALL OF THESE:**

```bash
# 1. Format code
uvx ruff format alea_llm_client/ tests/

# 2. Lint and fix issues
uvx ruff check --fix alea_llm_client/ tests/

# 3. Type check
uv run uvx ty check alea_llm_client/

# 4. Run all tests
uv run pytest

# 5. Run tests with coverage
uv run pytest --cov=alea_llm_client --cov-report=term-missing
```

**All of these must pass with zero errors before any code is considered complete.**

## Common Commands

```bash
# Set up development environment
uv sync

# Run a specific model test
uv run python -c "
from alea_llm_client.llms.models.anthropic_model import AnthropicModel
model = AnthropicModel()
response = model.chat('Hello')
print(response.text)
"

# Test all models quickly
uv run python -c "
from alea_llm_client.llms.models import *
# Test each model here
"

# Check if dependencies are correct (should only show httpx and pydantic)
uv run python -c "
import sys
import pkg_resources
runtime_deps = [pkg.project_name for pkg in pkg_resources.working_set 
                if not pkg.project_name.startswith(('pip', 'setuptools', 'wheel'))]
print('Runtime dependencies:', sorted(runtime_deps))
"
```

## Error Handling Patterns

All models should raise appropriate ALEA exceptions:

```python
from alea_llm_client.core.exceptions import (
    ALEAAuthenticationError,  # For 401/API key issues
    ALEAModelError,          # For model/API errors
    ALEARetryExhaustedError  # For retry failures
)
```

## Repository Hygiene

### Commit Message Standard

**ALWAYS** use [Conventional Commits](https://www.conventionalcommits.org/) format:

- `feat(models): add new Google Vertex AI integration`
- `fix(anthropic): handle rate limiting properly`
- `test(openai): add comprehensive async tests`
- `docs(readme): update installation instructions`

### Before Every Commit

```bash
# Run the full QA checklist
uvx ruff format alea_llm_client/ tests/
uvx ruff check --fix alea_llm_client/ tests/
uv run uvx ty check alea_llm_client/
uv run pytest --cov=alea_llm_client
```

**NEVER COMMIT CODE THAT FAILS ANY OF THESE CHECKS.**