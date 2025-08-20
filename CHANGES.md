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
