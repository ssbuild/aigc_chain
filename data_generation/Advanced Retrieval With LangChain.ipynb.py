# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/11/27 15:20
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


loader = DirectoryLoader('../data/PaulGrahamEssaysLarge/', glob="**/*.txt", show_progress=True)

docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
splits = text_splitter.split_documents(docs)

print (f"Your {len(docs)} documents have been split into {len(splits)} chunks")


if 'vectordb' in globals(): # If you've already made your vectordb this will delete it so you start fresh
    vectordb.delete_collection()

embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=splits, embedding=embedding)