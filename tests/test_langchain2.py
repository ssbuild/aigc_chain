# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/11/27 13:32

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    ChatPromptTemplate

OPENAI_API_KEY="112233"
openai_api_base="http://192.168.2.180:8081/v1"
model_name="qwen-chat-7b-int4"

chat = ChatOpenAI(model_name=model_name,openai_api_base=openai_api_base,openai_api_key=OPENAI_API_KEY,
                  streaming=True,
                  callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                  verbose=True, temperature=0)


prompt=PromptTemplate(
    template="Propose creative ways to incorporate {food_1} and {food_2} in the cuisine of the users choice.",
    input_variables=["food_1", "food_2"]
)

system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)


human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)


chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

chat_prompt_with_values = chat_prompt.format_prompt(food_1="Bacon",
                                                   food_2="Shrimp",
                                                   text="I really like food from Germany.")

resp = chat(chat_prompt_with_values.to_messages())

print(resp.content)