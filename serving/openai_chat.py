from fastapi.responses import StreamingResponse
from typing import List
import openai
from server.utils import get_model_worker_config, fschat_openai_api_address
from pydantic import BaseModel


class OpenAiMessage(BaseModel):
    role: str = "user"
    content: str = "hello"


class OpenAiChatMsgIn(BaseModel):
    model: str
    messages: List[OpenAiMessage]
    temperature: float = 0.7
    n: int = 1
    max_tokens: int = 1024
    stop: List[str] = []
    stream: bool = False
    presence_penalty: int = 0
    frequency_penalty: int = 0


async def openai_chat(msg: OpenAiChatMsgIn):

    print(msg)

    async def get_response(msg):
        data = msg.dict()

        try:
            response = await openai.ChatCompletion.acreate(**data)
            if msg.stream:
                async for data in response:
                    if choices := data.choices:
                        if chunk := choices[0].get("delta", {}).get("content"):
                            print(chunk, end="", flush=True)
                            yield chunk
            else:
                if response.choices:
                    answer = response.choices[0].message.content
                    print(answer)
                    yield(answer)
        except Exception as e:
            msg = f"获取ChatCompletion时出错：{e}"
            print(f'{e.__class__.__name__}: {msg}')

    return StreamingResponse(
        get_response(msg),
        media_type='text/event-stream',
    )
