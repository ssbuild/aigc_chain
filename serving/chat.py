from dataclasses import Field

from fastapi import Body
from fastapi.responses import StreamingResponse

from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, Union, Tuple, Dict, Callable, Any, Awaitable
import asyncio

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, ChatMessagePromptTemplate
from typing import List

from pydantic import BaseModel

from serving.openai_chat import OpenAiMessage

class History(BaseModel):
    """
    对话历史
    可从dict生成，如
    h = History(**{"role":"user","content":"你好"})
    也可转换为tuple，如
    h.to_msy_tuple = ("human", "你好")
    """
    role: str = Field(...)
    content: str = Field(...)

    def to_msg_tuple(self):
        return "ai" if self.role=="assistant" else "human", self.content

    def to_msg_template(self, is_raw=True) -> ChatMessagePromptTemplate:
        role_maps = {
            "ai": "assistant",
            "human": "user",
        }
        role = role_maps.get(self.role, self.role)
        if is_raw: # 当前默认历史消息都是没有input_variable的文本。
            content = "{% raw %}" + self.content + "{% endraw %}"
        else:
            content = self.content

        return ChatMessagePromptTemplate.from_template(
            content,
            "jinja2",
            role=role,
        )

    @classmethod
    def from_data(cls, h: Union[List, Tuple, Dict]) -> "History":
        if isinstance(h, (list,tuple)) and len(h) >= 2:
            h = cls(role=h[0], content=h[1])
        elif isinstance(h, dict):
            h = cls(**h)

        return h


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


def get_prompt_template(x):
    return x


def get_ChatOpenAI(
        model_name: str,
        temperature: float,
        streaming: bool = True,
        callbacks: List[Callable] = [],
        verbose: bool = True,
        **kwargs: Any,
) -> ChatOpenAI:
    model = ChatOpenAI(
        streaming=streaming,
        verbose=verbose,
        callbacks=callbacks,

        model_name=model_name,
        temperature=temperature,
        **kwargs
    )
    return model
async def wrap_done(fn: Awaitable, event: asyncio.Event):
    """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
    try:
        await fn
    except Exception as e:
        # TODO: handle exception
        msg = f"Caught exception: {e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
    finally:
        # Signal the aiter to stop.
        event.set()

async def chat(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
                history: List[History] = Body([],
                                       description="历史对话",
                                       examples=[[
                                           {"role": "user", "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                           {"role": "assistant", "content": "虎头虎脑"}]]
                                       ),
                stream: bool = Body(False, description="流式输出"),
                model_name: str = Body(LLM_MODEL, description="LLM 模型名称。"),
                temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                # top_p: float = Body(TOP_P, description="LLM 核采样。勿与temperature同时设置", gt=0.0, lt=1.0),
                prompt_name: str = Body("llm_chat", description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
         ):
    history = [History.from_data(h) for h in history]

    async def chat_iterator(query: str,
                            history: List[History] = [],
                            model_name: str = LLM_MODEL,
                            prompt_name: str = prompt_name,
                            ) -> AsyncIterable[str]:
        callback = AsyncIteratorCallbackHandler()
        model = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            callbacks=[callback],
        )

        prompt_template = get_prompt_template(prompt_name)
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])
        chain = LLMChain(prompt=chat_prompt, llm=model)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"input": query}),
            callback.done),
        )

        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield token
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield answer

        await task

    return StreamingResponse(chat_iterator(query=query,
                                           history=history,
                                           model_name=model_name,
                                           prompt_name=prompt_name),
                             media_type="text/event-stream")
