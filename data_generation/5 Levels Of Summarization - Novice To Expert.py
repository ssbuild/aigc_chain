# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/11/27 14:57
# Summarization is a fundamental building block of many LLM tasks. You'll frequently run into use cases where you would like to distill a large body of text into a succinct set of points.
#
# Depending on the length of the text you'd like to summarize, you have different summarization methods to choose from.
#
# We're going to run through 5 methods for summarization that start with Novice and end up expert. These aren't the only options, feel free to make up your own. If you find another one you like please share it with the community.
#
# 5 Levels Of Summarization:
#
# Summarize a couple sentences - Basic Prompt
# Summarize a couple paragraphs - Prompt Templates
# Summarize a couple pages - Map Reduce
# Summarize an entire book - Best Representation Vectors
# Summarize an unknown amount of text - Agents
# First let's import our OpenAI API Key


import os

model_args = dict(
    openai_api_key="112233",
    openai_api_base="http://192.168.2.180:8081/v1",
    model_name="qwen-chat-7b-int4",
)

from langchain.llms import OpenAI

llm = OpenAI(temperature=0, **model_args)

prompt = """
Please provide a summary of the following text

TEXT:
Philosophy (from Greek: φιλοσοφία, philosophia, 'love of wisdom') \
is the systematized study of general and fundamental questions, \
such as those about existence, reason, knowledge, values, mind, and language. \
Some sources claim the term was coined by Pythagoras (c. 570 – c. 495 BCE), \
although this theory is disputed by some. Philosophical methods include questioning, \
critical discussion, rational argument, and systematic presentation.
"""

num_tokens = llm.get_num_tokens(prompt)
print (f"Our prompt has {num_tokens} tokens")


output = llm(prompt)
print (output)



prompt = """
Please provide a summary of the following text.
Please provide your output in a manner that a 5 year old would understand

TEXT:
Philosophy (from Greek: φιλοσοφία, philosophia, 'love of wisdom') \
is the systematized study of general and fundamental questions, \
such as those about existence, reason, knowledge, values, mind, and language. \
Some sources claim the term was coined by Pythagoras (c. 570 – c. 495 BCE), \
although this theory is disputed by some. Philosophical methods include questioning, \
critical discussion, rational argument, and systematic presentation.
"""

num_tokens = llm.get_num_tokens(prompt)
print (f"Our prompt has {num_tokens} tokens")

output = llm(prompt)
print (output)
