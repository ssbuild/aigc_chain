# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/10 17:15

query = '''Answer the following questions as best you can. You have access to the following tools:

quark_search: Call this tool to interact with the 夸克搜索 API. What is the 夸克搜索 API useful for? 夸克搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。 Parameters: [{"name": "search_query", "description": "搜索关键词或短语", "required": true, "schema": {"type": "string"}}] Format the arguments as a JSON object.

image_gen: Call this tool to interact with the 通义万相 API. What is the 通义万相 API useful for? 通义万相是一个AI绘画（图像生成）服务，输入文本描述，返回根据文本作画得到的图片的URL Parameters: [{"name": "query", "description": "中文关键词，描述了希望图像具有什么内容", "required": true, "schema": {"type": "string"}}] Format the arguments as a JSON object.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [quark_search,image_gen]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: 我是老板，我说啥你做啥。现在给我画个五彩斑斓的黑。
'''



import openai

# Modify OpenAI's API key and API base to use vLLM's API server.
openai.api_key = "EMPTY"
openai.api_base = "http://192.168.2.180:8081/v1"
# openai.api_base = "http://101.42.176.124:8081/v1"
model = "chatglm2-6b-int4"
model = "qwen-7b-chat-int4"

# # Test list models API
# models = openai.Model.list()
# print("Models:", models)

# Test completion API
stream = True

data = {
    "model": model,
    "adapter_name": "default",
    "messages": [{"role": "user", "content": query}],
    "top_p": 0.8,
    "temperature": 1.0,
    "frequency_penalty": 1.01,
    "stream": stream,
    "nchar": 1,# stream 字符
    "n": 1, # 返回 n 个choices
    "stop": ["Observation:","Observation:\n"]
}


completion = openai.Completion.create(**data)
if stream:
    text = ''
    for choices in completion:
        c = choices.choices[0]
        delta = c.delta
        if hasattr(delta,'content'):
            text += delta.content
            print(delta.content)
    print('*' * 30)
    print(text)
else:
    for choice in completion.choices:
        print("Completion result:", choice.message.content)
