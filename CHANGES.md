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
