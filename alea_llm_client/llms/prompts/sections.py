# imports
import json
from typing import Any

# packages
from pydantic import BaseModel


def normalize_section_title(title: str) -> str:
    """
    Normalize a section title.

    Args:
    - title (str): the title to normalize

    Returns:
    - str: the normalized title
    """
    return title.strip().title()


def get_section_tag(title: str) -> str:
    """
    Get a valid XML tag name from the title, dealing with
    whitespace or other invalid XML tag characters.

    Args:
    - title (str): the title to convert to a tag

    Returns:
    - str: the converted tag
    """
    return title.strip().replace(" ", "-").upper()


def format_section_content(content: Any) -> str:
    """
    Format the section content by checking type and handling accordingly.
      - If str or bytes, return as is.
      - If list or dict, return as JSON string.
      - If a Pydantic model, return with model_dump()

    Args:
    - content (Any): the content to format

    Returns:
    - str: the formatted content
    """
    if isinstance(content, (str, bytes)):
        return content
    elif isinstance(content, (list, dict)):
        return json.dumps(content, default=str)
    elif isinstance(content, BaseModel):
        return content.model_dump_json()
    elif hasattr(content, "model_dump_json"):
        return json.dumps(content.model_json_schema())
    else:
        return str(content)


def format_instructions(instructions: list[str]) -> str:
    """
    Format instructions for a prompt.

    Args:
    - instructions (list[str]): a list of instructions to format

    Returns:
    - str: the formatted instructions
    """
    formatted_instructions = ""
    for number, instruction in enumerate(instructions, start=1):
        # strip and ensure we have a period at the end
        instruction = instruction.strip()
        if not instruction.endswith("."):
            instruction += "."
        formatted_instructions += f" - {number}. {instruction}\n"
    return formatted_instructions


def format_prompt_sections(sections: dict) -> list[str]:
    """
    Format prompt sections.

    Args:
    - sections (dict): the sections to format

    Returns:
    - list[str]: the formatted sections
    """
    formatted_sections = []
    for section_title, section_content in sections.items():
        section_title = normalize_section_title(section_title)
        section_tag = get_section_tag(section_title)
        formatted_sections.append(
            f"# {section_title}\n"
            f"<{section_tag}>\n"
            f"{format_section_content(section_content).strip()}\n"
            f"</{section_tag}>"
        )
    return formatted_sections


def format_prompt(sections: dict) -> str:
    """
    Format a prompt from a list of sections.

    Args:
    - sections (dict): the sections to format

    Returns:
    - str: the formatted prompt
    """
    formatted_sections = format_prompt_sections(sections)
    return "\n\n".join(formatted_sections)
