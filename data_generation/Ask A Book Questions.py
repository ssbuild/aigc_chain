#!/usr/bin/env python
# coding: utf-8

# # Top-K Similarity Search - Ask A Book A Question
# 
# In this tutorial we will see a simple example of basic retrieval via Top-K Similarity search

# In[1]:


# pip install langchain --upgrade
# Version: 0.0.164

# !pip install pypdf


# In[2]:


# PDF Loaders. If unstructured gives you a hard time, try PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()


# ### Load your data
# 
# Next let's load up some data. I've put a few 'loaders' on there which will load data from different locations. Feel free to use the one that suits you. The default one queries one of Paul Graham's essays for a simple example. This process will only stage the loader, not actually load it.

# In[3]:


loader = TextLoader(file_path="../data/PaulGrahamEssays/vb.txt")

## Other options for loaders 
# loader = PyPDFLoader("../data/field-guide-to-data-science.pdf")
# loader = UnstructuredPDFLoader("../data/field-guide-to-data-science.pdf")
# loader = OnlinePDFLoader("https://wolfpaulus.com/wp-content/uploads/2017/05/field-guide-to-data-science.pdf")


# Then let's go ahead and actually load the data.

# In[4]:


data = loader.load()


# Then let's actually check out what's been loaded

# In[5]:


# Note: If you're using PyPDFLoader then it will split by page for you already
print (f'You have {len(data)} document(s) in your data')
print (f'There are {len(data[0].page_content)} characters in your sample document')
print (f'Here is a sample: {data[0].page_content[:200]}')


# ### Chunk your data up into smaller documents
# 
# While we could pass the entire essay to a model w/ long context, we want to be picky about which information we share with our model. The better signal to noise ratio we have the more likely we are to get the right answer.
# 
# The first thing we'll do is chunk up our document into smaller pieces. The goal will be to take only a few of those smaller pieces and pass them to the LLM.

# In[6]:


# We'll split our data into chunks around 500 characters each with a 50 character overlap. These are relatively small.

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(data)


# In[7]:


# Let's see how many small chunks we have
print (f'Now you have {len(texts)} documents')


# ### Create embeddings of your documents to get ready for semantic search
# 
# Next up we need to prepare for similarity searches. The way we do this is through embedding our documents (getting a vector per document).
# 
# This will help us compare documents later on.

# In[8]:


from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone


# Check to see if there is an environment variable with you API keys, if not, use what you put below

# In[9]:


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YourAPIKey')


# Then we'll get our embeddings engine going. You can use whatever embeddings engine you would like. We'll use OpenAI's ada today.

# In[10]:


embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


# ### Option #1: Chroma (for local)
# 
# I like Chroma becauase it's local and easy to set up without an account.
# 
# First we'll pass our texts to Chroma via `.from_documents`, this will 1) embed the documents and get a vector, then 2) add them to the vectorstore for retrieval later.

# In[11]:


# load it into Chroma
vectorstore = Chroma.from_documents(texts, embeddings)


# Let's test it out. I want to see which documents are most closely related to a query.
# 
# 

# In[12]:


query = "What is great about having kids?"
docs = vectorstore.similarity_search(query)


# Then we can check them out. In theory, the texts which are deemed most similar should hold the answer to our question.
# But keep in mind that our query just happens to be a question, it could be a random statement or sentence and it would still work.

# In[13]:


# Here's an example of the first document that was returned
for doc in docs:
    print (f"{doc.page_content}\n")


# ### Option #2: Pinecone (for cloud)
# If you want to use pinecone, run the code below, if not then skip over to Chroma below it. You must go to [Pinecone.io](https://www.pinecone.io/) and set up an account

# In[14]:


# PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', 'YourAPIKey')
# PINECONE_API_ENV = os.getenv('PINECONE_API_ENV', 'us-east1-gcp') # You may need to switch with your env

# # initialize pinecone
# pinecone.init(
#     api_key=PINECONE_API_KEY,  # find at app.pinecone.io
#     environment=PINECONE_API_ENV  # next to api key in console
# )
# index_name = "langchaintest" # put in the name of your pinecone index here

# docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)


# ### Query those docs to get your answer back
# 
# Great, those are just the docs which should hold our answer. Now we can pass those to a LangChain chain to query the LLM.
# 
# We could do this manually, but a chain is a convenient helper for us.

# In[15]:


from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain


# In[16]:


llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")


# In[17]:


query = "What is great about having kids?"
docs = vectorstore.similarity_search(query)


# In[18]:


chain.run(input_documents=docs, question=query)


# Awesome! We just went and queried an external data source!
