# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/1 17:49
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
  id="player",
  description=(
      "用户正在控制音乐播放器来选择歌曲、暂停或启动它们或播放，特定艺术家的音乐"
  ),
  attributes=[
      Text(
          id="song",
          description="用户想要播放这首歌",
          examples=[],
          many=True,
      ),
      Text(
          id="album",
          description="用户想播放这张专辑",
          examples=[],
          many=True,
      ),
      Text(
          id="artist",
          description="给定艺术家的音乐",
          examples=[("保罗·西蒙的歌曲", "保罗·西蒙")],
          many=True,
      ),
      Text(
          id="action",
          description="采取以下操作之一：`播放`、`暂停`、`下一首`、`上一首`。",
          examples=[
              ("请停止音乐", "播放"),
              ("随机播放", "暂停"),
              ("播放一首歌", "暂停"),
              ("下一首歌曲", "下一首"),
          ],
      ),
    ],
  many=False,
)

chain = create_extraction_chain(llm, schema, encoder_or_encoder_class='json')

ret = chain.predict_and_parse(text="《我的地盘》是周杰伦演唱的一首歌曲，由周杰伦作曲，方文山作词，洪敬尧编曲，收录在周杰伦2004年8月3日发行的专辑《七里香》中 ")['data']

print(ret)