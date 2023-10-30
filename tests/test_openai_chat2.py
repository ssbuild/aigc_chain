# -*- coding: utf-8 -*-
import openai

# 新版本
openai.api_key = "EMPTY"
openai.api_base = "http://192.168.2.180:8081/v1"
# model = "chatglm2-6b-int4"
# model = "qwen-7b-chat-int4"
# model = "qwen-7b-chat"
model = "chatglm3-6b"
# # Test list models API
# models = openai.Model.list()
# print("Models:", models)
# Test completion API
stream = False

input_text ="以下是中国关于刑事诉讼法考试的单项选择题，请选出其中的正确答案。\n\n某市人民检察院在侦查该市财政局副局长张某受贿案的过程中，除发现张某利用职权之便收受他人贿赂之外，还发现张某涉嫌巨额财产来源不明罪和非法向外国人出售、赠送珍贵文物罪。指控张某犯有巨额财产来源不明罪，首先应当承担证明其财产或者支出明显超过其合法收入且差额巨大的证明责任的是下列哪个机关或人员?\nA. 检察院报人\nC. 张某本人\nD. 财政局\n答案："
input_text = "你是谁"
data = {
    "model": model,
    "adapter_name": None, # lora头
    "messages": [{"role": "user", "content": input_text}],
    "top_p": 1.0,
    "temperature": 0,
    "frequency_penalty": 1.01,
    "stream": stream,
    "nchar": 1,# stream 字符
    "n": 1, # 返回 n 个choices
    # "stop": ["Observation:"]
    "top_k": 1,
}

for i in range(1):
    completion = openai.ChatCompletion.create(**data)
    if stream:
        text = ''
        for choices in completion:
            c = choices.choices[0]
            delta = c.delta
            if hasattr(delta,'content'):
                text += delta.content
                print(delta.content)
        print(text)
    else:
        for choice in completion.choices:
            print("result:", choice.message.content)
