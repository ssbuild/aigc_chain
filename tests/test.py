# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/7/28 15:18
# import openai
# openai.api_key = "EMPTY"
# openai.api_base = "http://192.168.2.180:8081/v1"

from langchain.chat_models import ChatOpenAI
from kor import create_extraction_chain, Object, Text


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
        "User is controlling a music player to select songs, pause or start them or play"
        " music by a particular artist."
    ),
    attributes=[
        Text(
            id="song",
            description="User wants to play this song",
            examples=[],
            many=True,
        ),
        Text(
            id="album",
            description="User wants to play this album",
            examples=[],
            many=True,
        ),
        Text(
            id="artist",
            description="Music by the given artist",
            examples=[("Songs by paul simon", "paul simon")],
            many=True,
        ),
        Text(
            id="action",
            description="Action to take one of: `play`, `stop`, `next`, `previous`.",
            examples=[
                ("Please stop the music", "stop"),
                ("play something", "play"),
                ("play a song", "play"),
                ("next song", "next"),
            ],
        ),
    ],
    many=False,
)

chain = create_extraction_chain(llm, schema, encoder_or_encoder_class='json')
chain.run("play songs by paul simon and led zeppelin and the doors")['data']