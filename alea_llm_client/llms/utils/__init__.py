"""
Utility functions for working with LLM responses.
"""

from .json import normalize_json_response, replace_jsons_refs_with_enum

__all__ = [
    "normalize_json_response",
    "replace_jsons_refs_with_enum",
]
