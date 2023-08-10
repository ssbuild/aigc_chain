# -*- coding: utf-8 -*-
# @Time:  18:47
# @Author: tk
# @File：react2

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
Thought: 我应该使用通义万相API来生成一张五彩斑斓的黑的图片。
Action: image_gen
Action Input: {"query": "五彩斑斓的黑"}
Observation: {"status_code": 200, "request_id": "3d894da2-0e26-9b7c-bd90-102e5250ae03", "code": null, "message": "", "output": {"task_id": "2befaa09-a8b3-4740-ada9-4d00c2758b05", "task_status": "SUCCEEDED", "results": [{"url": "https://dashscope-result-sh.oss-cn-shanghai.aliyuncs.com/1e5e2015/20230801/1509/6b26bb83-469e-4c70-bff4-a9edd1e584f3-1.png"}], "task_metrics": {"TOTAL": 1, "SUCCEEDED": 1, "FAILED": 0}}, "usage": {"image_count": 1}}
'''

import openai

# Modify OpenAI's API key and API base to use vLLM's API server.
openai.api_key = "EMPTY"
# openai.api_base = "http://192.168.2.180:8081/v1"
openai.api_base = "http://101.42.176.124:8081/v1"
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
    # "stop": ["Observation:","Observation:\n"]
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