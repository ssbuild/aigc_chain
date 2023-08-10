# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/7/28 16:20
from kor import create_extraction_chain, Object, Text
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(
    # model_name="gpt-3.5-turbo",
    model_name="chatglm2-6b-int4",
    temperature=0,
    max_tokens=2000,
    frequency_penalty=0,
    presence_penalty=0,
    top_p=1.0,
    openai_api_key = "EMPTY",
    openai_api_base = "http://192.168.2.180:8081/v1"
)

schema = Object(
    id="person",
    description="Personal information",
    examples=[
        ("Alice and Bob are friends", [{"first_name": "Alice"}, {"first_name": "Bob"}])
    ],
    attributes=[
        Text(
            id="first_name",
            description="The first name of a person.",
        )
    ],
    many=True,
)

instruction_template = PromptTemplate(
    input_variables=["format_instructions", "type_description"],
    template=(
        "[Pep talk for your LLM goes here]\n\n"
        "Add some type description\n\n"
        "{type_description}\n\n" # Can comment out
        "Add some format instructions\n\n"
        "{format_instructions}\n"
        "Suffix heren\n"
    ),
)


chain = create_extraction_chain(llm, schema, instruction_template=instruction_template)

print(chain.prompt.format_prompt(text='hello').to_string())



