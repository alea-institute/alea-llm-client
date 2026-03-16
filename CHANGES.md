## Version 0.3.0

### New Models
* **OpenAI**: Added GPT-5.4 (1.05M context, 128K output), GPT-5.3, GPT-5.2 (400K context), GPT-5.1, plus codex and pro variants — 16 new models total
* **Anthropic**: Added Claude Opus 4.6 (1M context, 128K output), Claude Sonnet 4.6 (1M context, 64K output), Claude 4.5 series (Opus, Sonnet, Haiku), and Haiku 4.5 — 8 new models
* **Google**: Added Gemini 3.1 Pro Preview, Gemini 3.1 Flash Lite Preview, Gemini 3 Flash Preview, Gemini 3 Pro Preview, Gemini 2.5 Flash Lite — 5 new models
* **xAI**: Added Grok 4.20 beta series (multi-agent, reasoning, non-reasoning), Grok 4.1 Fast, Grok 4 Fast (reasoning/non-reasoning), Grok Code Fast 1, Grok 4-0709 — 9 new models

### New API Parameters
* **OpenAI `service_tier`**: Validate and pass `auto`, `default`, `flex`, `scale`, `priority`
* **OpenAI `reasoning_effort`**: Expanded from 4 to 6 values — added `"none"` and `"xhigh"`
* **Anthropic `service_tier`**: Validate and pass `auto`, `standard_only`
* **Anthropic `output_config`**: Control output effort (`low`, `medium`, `high`, `max`) and structured output format
* **Anthropic `metadata`**: Pass request metadata (user_id, etc.)
* **Google `thinking_level`**: Control thinking token budget (`minimal`, `low`, `medium`, `high`)
* **`max_tokens` auto-conversion**: Automatically converts `max_tokens` to `max_completion_tokens` for GPT-5.x and o-series models that require it

### Tool Creation Helpers
* **OpenAI Responses API**: 12 helpers — `create_web_search_tool`, `create_file_search_tool`, `create_code_interpreter_tool`, `create_computer_tool`, `create_mcp_tool`, `create_function_tool`, `create_image_generation_tool`, `create_local_shell_tool`, `create_tool_search_tool`, `create_apply_patch_tool`, and more
* **Anthropic Messages API**: 10 helpers — `create_anthropic_web_search_tool`, `create_anthropic_web_fetch_tool`, `create_anthropic_code_execution_tool`, `create_anthropic_bash_tool`, `create_anthropic_text_editor_tool`, `create_anthropic_computer_use_tool`, `create_anthropic_memory_tool`, `create_anthropic_tool_search_bm25_tool`, `create_anthropic_tool_search_regex_tool`, `create_anthropic_function_tool`

### Deprecation Handling
* **Anthropic deprecated models**: Claude 3.5, 3.7, and 3-Opus models now emit `DeprecationWarning` at construction with replacement suggestions
* **Clear 404 errors**: Retired models return `ALEAModelError` with "deprecated, use X instead" message instead of generic HTTP errors
* **`DEPRECATED_MODEL_REPLACEMENTS` mapping**: Programmatic access via `is_deprecated()` and `get_model_replacement()`

### Response Handling Improvements
* **OpenAI Responses API**: Handle new tool output types — `web_search_call`, `file_search_call`, `code_interpreter_call`, `computer_call`, `mcp_call`, `image_generation_call`, `local_shell_call`, `function_call`
* **Anthropic `service_tier`**: Extracted from response into `metadata["service_tier"]`
* **Google thinking tokens**: Extract `thoughtsTokenCount` from usage into `ModelResponse.reasoning_tokens`
* **Google empty parts fix**: Handle candidates with empty parts without crashing
* **xAI error parsing**: Fix `_extract_error_message` to handle xAI's string error format (`{"error": "string"}`) vs OpenAI's nested format

### Updated Defaults
* **OpenAI**: `gpt-5.4` (was `gpt-5-chat-latest`)
* **Anthropic**: `claude-sonnet-4-6` (was `claude-sonnet-4-20250514`)
* **Google**: `gemini-3.1-pro-preview` (was `gemini-2.0-flash-exp`)
* **xAI**: `grok-4-fast-non-reasoning` (was `grok-2-1212`)

### Testing
* **196 integration tests** across all providers, 71% code coverage
* **49 OpenAI tests**, **57 Anthropic tests**, **24 Google tests**, **22 Grok tests**, **10 model capability tests**, **10 model registry tests**, **24 grammar tests**
* All tests use real API calls with rate limiting — no mocks

### Bug Fixes
* **Google `gemini-2.0-flash-exp` retired**: Updated default and fixture models
* **Grok `grok-2-1212` retired**: Updated default model
* **Anthropic Claude 3.5/3.7 retired**: Added deprecation warnings and clear error messages
* **xAI error parsing crash**: Fixed `'str' object has no attribute 'get'` when xAI returns string errors
* **Model spec corrections**: Fixed context windows and max output tokens based on web search verification (GPT-5.4: 1.05M/128K, Claude Opus 4.6: 1M/128K, GPT-5.2: 400K/128K)

## Version 0.2.3

### Architecture & Performance Improvements
* **Eliminated Circular Imports**: Completely refactored dynamic loading system to use proper Python import patterns
  - Removed all lazy imports and `PLC0415` violations 
  - Moved model categorization logic to `ModelConfigLoader` class methods
  - Implemented top-level imports with robust fallback patterns
  - Enhanced model registry stability and maintainability

### Model Registry Enhancements  
* **Complete OpenAI Model Coverage**: Updated to include all latest OpenAI models from official API documentation
  - Added o-series models: `o4-mini`, `o3-deep-research`, `o4-mini-deep-research` with date variants
  - Added specialized models: `computer-use-preview`, audio, search, and realtime model variants
  - Added GPT-4.1 family: `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`
  - Enhanced fallback model definitions for 100% API compatibility

### Testing Improvements
* **Fixed All Model Categorization Tests**: Resolved dynamic loading failures that caused empty model lists
  - Fixed fallback logic to properly detect when dynamic loading returns empty results
  - Enhanced GPT-4.1 categorization order precedence (before GPT-4 matching)
  - Improved Sonnet/Opus/Haiku model name matching patterns
* **Fixed All Cache Functionality Tests**: Resolved cache test failures across all providers 
  - Updated cache default behavior (`ignore_cache=True`) with proper test configurations
  - Added explicit `ignore_cache=False` settings in all cache test functions
  - Fixed syntax errors from automated replacements in test files
* **Grammar Functionality Testing**: Complete validation of GPT-5 grammar constraint features
  - All 24 grammar tests passing with proper API authentication
  - Verified Lark and regex grammar syntax validation
  - Confirmed grammar constraints work exclusively with GPT-5 models

### Bug Fixes
* **Test Syntax Errors**: Fixed invalid Python syntax created by automated text replacements (`.*.ignore_cache`)
* **Dynamic Model Loading**: Fixed empty model list fallback detection and category assignment
* **Model Categorization Logic**: Fixed precedence issues in model family detection and naming patterns

### Quality Assurance
* **Complete Test Coverage**: Achieved 100% test pass rate across all providers and features
  - 178 total tests passing (previously 165 passed, 13 failed)
  - All model providers: Anthropic, OpenAI, Google, Grok, VLLM
  - All advanced features: Grammar constraints, caching, model categorization
* **Code Quality**: Reduced linting issues to only intentional architectural decisions
  - Fixed whitespace and formatting issues throughout codebase
  - Enhanced type safety and error handling

## Version 0.2.2

### Breaking Changes (Improved UX)
* **Caching Disabled by Default**: Changed default `ignore_cache=False` to `ignore_cache=True` for more predictable API client behavior
  - Users now get fresh responses by default unless they explicitly enable caching
  - To enable caching: `model = OpenAIModel(ignore_cache=False)` or pass `ignore_cache=False` to individual method calls
  - More consistent with typical API client expectations

### API Cleanup & Simplification  
* **Simplified Prompt Formatting**: Consolidated confusing prompt formatting exports to clean public API
  - Kept: `format_prompt` and `format_instructions` (used in examples and documentation)
  - Removed from public API: `format_model_prompt`, `format_prompt_llama3`, `format_prompt_sections`, `format_section_content`, `TokenType`
  - Internal formatting functions are still available via direct module imports if needed

### Development & Maintenance Improvements
* **Consolidated Linting**: Migrated from pylint + ruff to ruff-only configuration for faster, simpler linting
  - Removed: All pylint configuration and dependencies
  - Enhanced: Comprehensive ruff configuration with pylint-equivalent rules
  - Faster development workflow with single linter
* **Repository Cleanup**: Removed development artifacts and unnecessary files for cleaner project structure

### Testing Improvements
* **Dynamic Model Testing**: Updated tests to rely on dynamic model loading rather than static enums
  - Tests now validate that real models are available from JSON configuration
  - More robust testing of the actual dynamic system users experience
  - Better coverage of provider model availability

## Version 0.2.1

### Major Infrastructure Improvements
* **Dynamic Model Registry**: Complete architectural overhaul from hardcoded enums to dynamic JSON-based model configuration sourced from real API calls
* **97 Real Models**: Expanded model support with live API data - 72 OpenAI models, 9 Anthropic models, 7 Google models, 9 xAI models
* **Real API Integration**: All model configurations now derived from actual provider API responses rather than documentation, ensuring accuracy and completeness

### Testing & Quality Improvements  
* **Comprehensive Rate Limiting**: Added intelligent rate limiting system in test infrastructure to prevent API quota exhaustion
  - Configurable delays per provider (Google: 2.0s, Anthropic: 0.5s, OpenAI: 0.2s, xAI: 1.0s, VLLM: 0.1s)
  - Concurrent request limits to respect provider constraints
  - Environment variable configuration for custom rate limits
* **Enhanced Test Infrastructure**: Fixed retry mechanism tests, cache functionality interference, and wrapper transparency issues
* **VLLM Server Detection**: Fixed environment variable detection for custom VLLM endpoints (VLLM_ENDPOINT, VLLM_MODEL)

### Developer Experience Enhancements
* **Transparent Rate Limiting**: Rate limiting wrapper provides seamless integration without affecting model functionality
* **Improved Error Handling**: Better error messages and retry logic for API failures
* **Enhanced Configuration**: Support for custom cache paths, endpoint detection, and model assignment

### Breaking Changes
* **Model Registry Format**: Model configuration moved from Python constants to JSON files (backward compatibility maintained through fallback constants)
* **Test Configuration**: Rate limiting may affect test execution times (can be customized via environment variables)

### Bug Fixes
* **Cache Interference**: Fixed cache functionality interfering with mock patching in tests
* **Wrapper Transparency**: Enhanced RateLimitedModel wrapper to properly forward all attribute operations
* **Retry Mechanism**: Fixed retry exhaustion tests requiring `ignore_cache=True` for proper mock interaction

## Version 0.2.0

### Major Features
* **Model Registry & Capability Matrix**: Added comprehensive model registry with detailed capability tracking for 50+ models across OpenAI, Anthropic, Google, and xAI
* **Advanced Model Querying**: Implemented sophisticated model filtering and comparison capabilities with functions like `get_models_with_context_window_gte()`, `filter_models()`, and `compare_models()`
* **Enhanced Anthropic Support**: Added thinking mode support for Claude 4+ models with configurable budget tokens
* **Enhanced OpenAI Support**: Added reasoning token tracking for o-series models and grammar constraint support for GPT-5 models
* **Advanced Response APIs**: Added `responses()` methods for structured output with tool use, grammar constraints, and reasoning modes

### Model Improvements
* **Google Model Fixes**: Fixed naming consistency (`~/.alea/keys/gemini` instead of `~/.alea/keys/google`), improved API endpoint detection for both Vertex AI and Generative AI, and fixed authentication handling
* **Type Safety**: Complete type system overhaul with proper generic Pydantic support and elimination of all type checking issues
* **Response Handling**: Enhanced `ModelResponse` with thinking content, reasoning tokens, and content blocks support

### API Enhancements
* **Grammar Constraints**: Added grammar constraint support for GPT-5 models with Lark and regex syntax validation
* **Thinking Mode**: Added thinking mode support for Claude models with budget token configuration
* **Reasoning Tokens**: Added reasoning token tracking for OpenAI o-series models
* **Structured Output**: Enhanced JSON and Pydantic response handling with better error messages

### Testing & Quality
* **Comprehensive Test Suite**: Added 400+ lines of new tests covering model capabilities, grammar constraints, and advanced features
* **Type Safety**: Fixed all 66 type checking issues with proper null safety and generic type support
* **Code Quality**: Improved error handling, logging, and documentation throughout

### Breaking Changes
* **Google Model**: Key file path changed from `~/.alea/keys/google` to `~/.alea/keys/gemini` for consistency
* **ModelResponse**: `text` field changed from `Optional[str]` to `str` with default empty string
* **Pydantic Methods**: Return types changed to use generics (`T` instead of `BaseModel`) for better type safety

## Version 0.1.3
* Updated version, dependencies, and documentation
* Added new models to package export

## Version 0.1.2
* Added Google Vertex API support
* Added Grok API support

## Version 0.1.1
* Refactored on base httpx/pydantic dependencies
