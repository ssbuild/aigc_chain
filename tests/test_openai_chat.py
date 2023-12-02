import openai

# 新版本
openai.api_key = "112233"
openai.api_base = "http://106.12.147.243:9090/v1"
model = "chatglm2-6b-int4"
model = "ChatYuan-large-v2"
model = "Qwen-14B-Chat"

# Test completion API
stream = False

data = {
    "model": model,
    "adapter_name": None, # lora头
    "messages": [{"role": "user", "content": "你是谁"}],
    "top_p": 0.8,
    "temperature": 1.0,
    "frequency_penalty": 1.1,
    "stream": stream,
    "nchar": 1,# stream 字符
    "n": 1, # 返回 n 个choices
    "max_tokens": 512,
    "stop": ["Observation:"],
    "seed": None,
    # "seed": 46,
}


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
