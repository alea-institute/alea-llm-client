from typing import cast

from pydantic import BaseModel

from alea_llm_client import AnthropicModel, OpenAIModel, VLLMModel
from alea_llm_client.llms.prompts.sections import format_instructions, format_prompt


class Person(BaseModel):
    name: str
    age: int


def test_pydantic_vllm():
    instructions = [
        "Provide one random record based on the SCHEMA below.",
    ]
    prompt = format_prompt(
        {
            "instructions": format_instructions(instructions),
            "schema": Person,
        }
    )

    model = VLLMModel()
    person = cast(Person, model.pydantic(prompt, system="Respond in JSON.", pydantic_model=Person))

    assert isinstance(person.name, str)
    assert isinstance(person.age, int)


def test_pydantic_openai():
    instructions = [
        "Provide one random record based on the SCHEMA below.",
    ]
    prompt = format_prompt(
        {
            "instructions": format_instructions(instructions),
            "schema": Person,
        }
    )

    model = OpenAIModel()
    person = cast(Person, model.pydantic(prompt, system="Respond in JSON.", pydantic_model=Person))

    assert isinstance(person.name, str)
    assert isinstance(person.age, int)


def test_pydantic_anthropic():
    instructions = [
        "Provide one random record based on the SCHEMA below.",
    ]
    prompt = format_prompt(
        {
            "instructions": format_instructions(instructions),
            "schema": Person,
        }
    )

    model = AnthropicModel()
    person = cast(Person, model.pydantic(prompt, system="Respond in JSON.", pydantic_model=Person))

    assert isinstance(person.name, str)
    assert isinstance(person.age, int)
