#!/usr/bin/env python
# coding: utf-8

# # LangChain Cookbook 👨‍🍳👩‍🍳

# *This cookbook is based off the [LangChain Conceptual Documentation](https://docs.langchain.com/docs/)*
# 
# **Goal:** Provide an introductory understanding of the components and use cases of LangChain via [ELI5](https://www.dictionary.com/e/slang/eli5/#:~:text=ELI5%20is%20short%20for%20%E2%80%9CExplain,a%20complicated%20question%20or%20problem.) examples and code snippets. For use cases check out [part 2](https://github.com/gkamradt/langchain-tutorials/blob/main/LangChain%20Cookbook%20Part%202%20-%20Use%20Cases.ipynb). See [video tutorial](https://www.youtube.com/watch?v=2xxziIWmaSA) of this notebook.
# 
# 
# **Links:**
# * [LC Conceptual Documentation](https://docs.langchain.com/docs/)
# * [LC Python Documentation](https://python.langchain.com/en/latest/)
# * [LC Javascript/Typescript Documentation](https://js.langchain.com/docs/)
# * [LC Discord](https://discord.gg/6adMQxSpJS)
# * [www.langchain.com](https://langchain.com/)
# * [LC Twitter](https://twitter.com/LangChainAI)
# 
# 
# ### **What is LangChain?**
# > LangChain is a framework for developing applications powered by language models.
# 
# **~~TL~~DR**: LangChain makes the complicated parts of working & building with AI models easier. It helps do this in two ways:
# 
# 1. **Integration** - Bring external data, such as your files, other applications, and api data, to your LLMs
# 2. **Agency** - Allow your LLMs to interact with it's environment via decision making. Use LLMs to help decide which action to take next
# 
# ### **Why LangChain?**
# 1. **Components** - LangChain makes it easy to swap out abstractions and components necessary to work with language models.
# 
# 2. **Customized Chains** - LangChain provides out of the box support for using and customizing 'chains' - a series of actions strung together.
# 
# 3. **Speed 🚢** - This team ships insanely fast. You'll be up to date with the latest LLM features.
# 
# 4. **Community 👥** - Wonderful discord and community support, meet ups, hackathons, etc.
# 
# Though LLMs can be straightforward (text-in, text-out) you'll quickly run into friction points that LangChain helps with once you develop more complicated applications.
# 
# *Note: This cookbook will not cover all aspects of LangChain. It's contents have been curated to get you to building & impact as quick as possible. For more, please check out [LangChain Conceptual Documentation](https://docs.langchain.com/docs/)*
# 
# *Update Oct '23: This notebook has been expanded from it's original form*
# 
# You'll need an OpenAI api key to follow this tutorial. You can have it as an environement variable, in an .env file where this jupyter notebook lives, or insert it below where 'YourAPIKey' is. Have if you have questions on this, put these instructions into [ChatGPT](https://chat.openai.com/).

# In[1]:


from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key=os.getenv('OPENAI_API_KEY', 'YourAPIKey')


# # LangChain Components
# 
# ## Schema - Nuts and Bolts of working with Large Language Models (LLMs)
# 
# ### **Text**
# The natural language way to interact with LLMs

# In[2]:


# You'll be working with simple strings (that'll soon grow in complexity!)
my_text = "What day comes after Friday?"
my_text


# ### **Chat Messages**
# Like text, but specified with a message type (System, Human, AI)
# 
# * **System** - Helpful background context that tell the AI what to do
# * **Human** - Messages that are intented to represent the user
# * **AI** - Messages that show what the AI responded with
# 
# For more, see OpenAI's [documentation](https://platform.openai.com/docs/guides/chat/introduction)

# In[3]:


from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# This it the language model we'll use. We'll talk about what we're doing below in the next section
chat = ChatOpenAI(temperature=.7, openai_api_key=openai_api_key)


# Now let's create a few messages that simulate a chat experience with a bot

# In[4]:


chat(
    [
        SystemMessage(content="You are a nice AI bot that helps a user figure out what to eat in one short sentence"),
        HumanMessage(content="I like tomatoes, what should I eat?")
    ]
)


# You can also pass more chat history w/ responses from the AI

# In[5]:


chat(
    [
        SystemMessage(content="You are a nice AI bot that helps a user figure out where to travel in one short sentence"),
        HumanMessage(content="I like the beaches where should I go?"),
        AIMessage(content="You should go to Nice, France"),
        HumanMessage(content="What else should I do when I'm there?")
    ]
)


# You can also exclude the system message if you want

# In[6]:


chat(
    [
        HumanMessage(content="What day comes after Thursday?")
    ]
)


# ### **Documents**
# An object that holds a piece of text and metadata (more information about that text)

# In[7]:


from langchain.schema import Document


# In[8]:


Document(page_content="This is my document. It is full of text that I've gathered from other places",
         metadata={
             'my_document_id' : 234234,
             'my_document_source' : "The LangChain Papers",
             'my_document_create_time' : 1680013019
         })


# But you don't have to include metadata if you don't want to

# In[9]:


Document(page_content="This is my document. It is full of text that I've gathered from other places")


# ## Models - The interface to the AI brains

# ###  **Language Model**
# A model that does text in ➡️ text out!
# 
# *Check out how I changed the model I was using from the default one to ada-001 (a very cheap, low performing model). See more models [here](https://platform.openai.com/docs/models)*

# In[10]:


from langchain.llms import OpenAI

llm = OpenAI(model_name="text-ada-001", openai_api_key=openai_api_key)


# In[11]:


llm("What day comes after Friday?")


# ### **Chat Model**
# A model that takes a series of messages and returns a message output

# In[12]:


from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

chat = ChatOpenAI(temperature=1, openai_api_key=openai_api_key)


# In[13]:


chat(
    [
        SystemMessage(content="You are an unhelpful AI bot that makes a joke at whatever the user says"),
        HumanMessage(content="I would like to go to New York, how should I do this?")
    ]
)


# ### Function Calling Models
# 
# [Function calling models](https://openai.com/blog/function-calling-and-other-api-updates) are similar to Chat Models but with a little extra flavor. They are fine tuned to give structured data outputs.
# 
# This comes in handy when you're making an API call to an external service or doing extraction.

# In[14]:


chat = ChatOpenAI(model='gpt-3.5-turbo-0613', temperature=1, openai_api_key=openai_api_key)

output = chat(messages=
     [
         SystemMessage(content="You are an helpful AI bot"),
         HumanMessage(content="What’s the weather like in Boston right now?")
     ],
     functions=[{
         "name": "get_current_weather",
         "description": "Get the current weather in a given location",
         "parameters": {
             "type": "object",
             "properties": {
                 "location": {
                     "type": "string",
                     "description": "The city and state, e.g. San Francisco, CA"
                 },
                 "unit": {
                     "type": "string",
                     "enum": ["celsius", "fahrenheit"]
                 }
             },
             "required": ["location"]
         }
     }
     ]
)
output


# See the extra `additional_kwargs` that is passed back to us? We can take that and pass it to an external API to get data. It saves the hassle of doing output parsing.

# ### **Text Embedding Model**
# Change your text into a vector (a series of numbers that hold the semantic 'meaning' of your text). Mainly used when comparing two pieces of text together.
# 
# *BTW: Semantic means 'relating to meaning in language or logic.'*

# In[15]:


from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)


# In[16]:


text = "Hi! It's time for the beach"


# In[17]:


text_embedding = embeddings.embed_query(text)
print (f"Here's a sample: {text_embedding[:5]}...")
print (f"Your embedding is length {len(text_embedding)}")


# ## Prompts - Text generally used as instructions to your model

# ### **Prompt**
# What you'll pass to the underlying model

# In[18]:


from langchain.llms import OpenAI

llm = OpenAI(model_name="text-davinci-003", openai_api_key=openai_api_key)

# I like to use three double quotation marks for my prompts because it's easier to read
prompt = """
Today is Monday, tomorrow is Wednesday.

What is wrong with that statement?
"""

print(llm(prompt))


# ### **Prompt Template**
# An object that helps create prompts based on a combination of user input, other non-static information and a fixed template string.
# 
# Think of it as an [f-string](https://realpython.com/python-f-strings/) in python but for prompts
# 
# *Advanced: Check out LangSmithHub(https://smith.langchain.com/hub) for many more communit prompt templates*

# In[19]:


from langchain.llms import OpenAI
from langchain import PromptTemplate

llm = OpenAI(model_name="text-davinci-003", openai_api_key=openai_api_key)

# Notice "location" below, that is a placeholder for another value later
template = """
I really want to travel to {location}. What should I do there?

Respond in one short sentence
"""

prompt = PromptTemplate(
    input_variables=["location"],
    template=template,
)

final_prompt = prompt.format(location='Rome')

print (f"Final Prompt: {final_prompt}")
print ("-----------")
print (f"LLM Output: {llm(final_prompt)}")


# ### **Example Selectors**
# An easy way to select from a series of examples that allow you to dynamic place in-context information into your prompt. Often used when your task is nuanced or you have a large list of examples.
# 
# Check out different types of example selectors [here](https://python.langchain.com/docs/modules/model_io/prompts/example_selectors/)
# 
# If you want an overview on why examples are important (prompt engineering), check out [this video](https://www.youtube.com/watch?v=dOxUroR57xs)

# In[20]:


from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.llms import OpenAI

llm = OpenAI(model_name="text-davinci-003", openai_api_key=openai_api_key)

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Example Input: {input}\nExample Output: {output}",
)

# Examples of locations that nouns are found
examples = [
    {"input": "pirate", "output": "ship"},
    {"input": "pilot", "output": "plane"},
    {"input": "driver", "output": "car"},
    {"input": "tree", "output": "ground"},
    {"input": "bird", "output": "nest"},
]


# In[21]:


# SemanticSimilarityExampleSelector will select examples that are similar to your input by semantic meaning

example_selector = SemanticSimilarityExampleSelector.from_examples(
    # This is the list of examples available to select from.
    examples, 
    
    # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
    OpenAIEmbeddings(openai_api_key=openai_api_key), 
    
    # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
    Chroma, 
    
    # This is the number of examples to produce.
    k=2
)


# In[22]:


similar_prompt = FewShotPromptTemplate(
    # The object that will help select examples
    example_selector=example_selector,
    
    # Your prompt
    example_prompt=example_prompt,
    
    # Customizations that will be added to the top and bottom of your prompt
    prefix="Give the location an item is usually found in",
    suffix="Input: {noun}\nOutput:",
    
    # What inputs your prompt will receive
    input_variables=["noun"],
)


# In[23]:


# Select a noun!
my_noun = "plant"
# my_noun = "student"

print(similar_prompt.format(noun=my_noun))


# In[24]:


llm(similar_prompt.format(noun=my_noun))


# ### **Output Parsers Method 1: Prompt Instructions & String Parsing**
# A helpful way to format the output of a model. Usually used for structured output. LangChain has a bunch more output parsers listed on their [documentation](https://python.langchain.com/docs/modules/model_io/output_parsers).
# 
# Two big concepts:
# 
# **1. Format Instructions** - A autogenerated prompt that tells the LLM how to format it's response based off your desired result
# 
# **2. Parser** - A method which will extract your model's text output into a desired structure (usually json)

# In[25]:


from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI


# In[26]:


llm = OpenAI(model_name="text-davinci-003", openai_api_key=openai_api_key)


# In[27]:


# How you would like your response structured. This is basically a fancy prompt template
response_schemas = [
    ResponseSchema(name="bad_string", description="This a poorly formatted user input string"),
    ResponseSchema(name="good_string", description="This is your response, a reformatted response")
]

# How you would like to parse your output
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)


# In[28]:


# See the prompt template you created for formatting
format_instructions = output_parser.get_format_instructions()
print (format_instructions)


# In[29]:


template = """
You will be given a poorly formatted string from a user.
Reformat it and make sure all the words are spelled correctly

{format_instructions}

% USER INPUT:
{user_input}

YOUR RESPONSE:
"""

prompt = PromptTemplate(
    input_variables=["user_input"],
    partial_variables={"format_instructions": format_instructions},
    template=template
)

promptValue = prompt.format(user_input="welcom to califonya!")

print(promptValue)


# In[30]:


llm_output = llm(promptValue)
llm_output


# In[31]:


output_parser.parse(llm_output)


# ### **Output Parsers Method 2: OpenAI Fuctions**
# When OpenAI released function calling, the game changed. This is recommended method when starting out.
# 
# They trained models specifically for outputing structured data. It became super easy to specify a Pydantic schema and get a structured output.
# 
# There are many ways to define your schema, I prefer using Pydantic Models because of how organized they are. Feel free to reference OpenAI's [documention](https://platform.openai.com/docs/guides/gpt/function-calling) for other methods.
# 
# In order to use this method you'll need to use a model that supports [function calling](https://openai.com/blog/function-calling-and-other-api-updates#:~:text=Developers%20can%20now%20describe%20functions%20to%20gpt%2D4%2D0613%20and%20gpt%2D3.5%2Dturbo%2D0613%2C). I'll use `gpt4-0613`
# 
# **Example 1: Simple**
# 
# Let's get started by defining a simple model for us to extract from.

# In[32]:


from langchain.pydantic_v1 import BaseModel, Field
from typing import Optional

class Person(BaseModel):
    """Identifying information about a person."""

    name: str = Field(..., description="The person's name")
    age: int = Field(..., description="The person's age")
    fav_food: Optional[str] = Field(None, description="The person's favorite food")


# Then let's create a chain (more on this later) that will do the extracting for us

# In[33]:


from langchain.chains.openai_functions import create_structured_output_chain

llm = ChatOpenAI(model='gpt-4-0613', openai_api_key=openai_api_key)

chain = create_structured_output_chain(Person, llm, prompt)
chain.run(
    "Sally is 13, Joey just turned 12 and loves spinach. Caroline is 10 years older than Sally."
)


# Notice how we only have data on one person from that list? That is because we didn't specify we wanted multiple. Let's change our schema to specify that we want a list of people if possible.

# In[34]:


from typing import Sequence

class People(BaseModel):
    """Identifying information about all people in a text."""

    people: Sequence[Person] = Field(..., description="The people in the text")


# Now we'll call for People rather than Person

# In[35]:


chain = create_structured_output_chain(People, llm, prompt)
chain.run(
    "Sally is 13, Joey just turned 12 and loves spinach. Caroline is 10 years older than Sally."
)


# Let's do some more parsing with it
# 
# **Example 2: Enum**
# 
# Now let's parse when a product from a list is mentioned

# In[36]:


import enum

llm = ChatOpenAI(model='gpt-4-0613', openai_api_key=openai_api_key)

class Product(str, enum.Enum):
    CRM = "CRM"
    VIDEO_EDITING = "VIDEO_EDITING"
    HARDWARE = "HARDWARE"


# In[37]:


class Products(BaseModel):
    """Identifying products that were mentioned in a text"""

    products: Sequence[Product] = Field(..., description="The products mentioned in a text")


# In[38]:


chain = create_structured_output_chain(Products, llm, prompt)
chain.run(
    "The CRM in this demo is great. Love the hardware. The microphone is also cool. Love the video editing"
)


# ## Indexes - Structuring documents to LLMs can work with them

# ### **Document Loaders**
# Easy ways to import data from other sources. Shared functionality with [OpenAI Plugins](https://openai.com/blog/chatgpt-plugins) [specifically retrieval plugins](https://github.com/openai/chatgpt-retrieval-plugin)
# 
# See a [big list](https://python.langchain.com/en/latest/modules/indexes/document_loaders.html) of document loaders here. A bunch more on [Llama Index](https://llamahub.ai/) as well.

# **HackerNews**

# In[39]:


from langchain.document_loaders import HNLoader


# In[40]:


loader = HNLoader("https://news.ycombinator.com/item?id=34422627")


# In[41]:


data = loader.load()


# In[42]:


print (f"Found {len(data)} comments")
print (f"Here's a sample:\n\n{''.join([x.page_content[:150] for x in data[:2]])}")


# **Books from Gutenberg Project**

# In[43]:


from langchain.document_loaders import GutenbergLoader

loader = GutenbergLoader("https://www.gutenberg.org/cache/epub/2148/pg2148.txt")

data = loader.load()


# In[44]:


print(data[0].page_content[1855:1984])


# **URLs and webpages**
# 
# Let's try it out with [Paul Graham's website](http://www.paulgraham.com/)

# In[45]:


from langchain.document_loaders import UnstructuredURLLoader

urls = [
    "http://www.paulgraham.com/",
]

loader = UnstructuredURLLoader(urls=urls)

data = loader.load()

data[0].page_content


# ### **Text Splitters**
# Often times your document is too long (like a book) for your LLM. You need to split it up into chunks. Text splitters help with this.
# 
# There are many ways you could split your text into chunks, experiment with [different ones](https://python.langchain.com/en/latest/modules/indexes/text_splitters.html) to see which is best for you.

# In[46]:


from langchain.text_splitter import RecursiveCharacterTextSplitter


# In[47]:


# This is a long document we can split up.
with open('data/PaulGrahamEssays/worked.txt') as f:
    pg_work = f.read()
    
print (f"You have {len([pg_work])} document")


# In[48]:


text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 150,
    chunk_overlap  = 20,
)

texts = text_splitter.create_documents([pg_work])


# In[49]:


print (f"You have {len(texts)} documents")


# In[50]:


print ("Preview:")
print (texts[0].page_content, "\n")
print (texts[1].page_content)


# There are a ton of different ways to do text splitting and it really depends on your retrieval strategy and application design. Check out more splitters [here](https://python.langchain.com/docs/modules/data_connection/document_transformers/)

# ### **Retrievers**
# Easy way to combine documents with language models.
# 
# There are many different types of retrievers, the most widely supported is the VectoreStoreRetriever

# In[51]:


from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

loader = TextLoader('data/PaulGrahamEssays/worked.txt')
documents = loader.load()


# In[52]:


# Get your splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

# Split your docs into texts
texts = text_splitter.split_documents(documents)

# Get embedding engine ready
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Embedd your texts
db = FAISS.from_documents(texts, embeddings)


# In[53]:


# Init your retriever. Asking for just 1 document back
retriever = db.as_retriever()


# In[54]:


retriever


# In[55]:


docs = retriever.get_relevant_documents("what types of things did the author want to build?")


# In[56]:


print("\n\n".join([x.page_content[:200] for x in docs[:2]]))


# ### **VectorStores**
# Databases to store vectors. Most popular ones are [Pinecone](https://www.pinecone.io/) & [Weaviate](https://weaviate.io/). More examples on OpenAIs [retriever documentation](https://github.com/openai/chatgpt-retrieval-plugin#choosing-a-vector-database). [Chroma](https://www.trychroma.com/) & [FAISS](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) are easy to work with locally.
# 
# Conceptually, think of them as tables w/ a column for embeddings (vectors) and a column for metadata.
# 
# Example
# 
# | Embedding      | Metadata |
# | ----------- | ----------- |
# | [-0.00015641732898075134, -0.003165106289088726, ...]      | {'date' : '1/2/23}       |
# | [-0.00035465431654651654, 1.4654131651654516546, ...]   | {'date' : '1/3/23}        |

# In[57]:


from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

loader = TextLoader('data/PaulGrahamEssays/worked.txt')
documents = loader.load()

# Get your splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

# Split your docs into texts
texts = text_splitter.split_documents(documents)

# Get embedding engine ready
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)


# In[58]:


print (f"You have {len(texts)} documents")


# In[59]:


embedding_list = embeddings.embed_documents([text.page_content for text in texts])


# In[60]:


print (f"You have {len(embedding_list)} embeddings")
print (f"Here's a sample of one: {embedding_list[0][:3]}...")


# Your vectorstore store your embeddings (☝️) and make them easily searchable

# ## Memory
# Helping LLMs remember information.
# 
# Memory is a bit of a loose term. It could be as simple as remembering information you've chatted about in the past or more complicated information retrieval.
# 
# We'll keep it towards the Chat Message use case. This would be used for chat bots.
# 
# There are many types of memory, explore [the documentation](https://python.langchain.com/en/latest/modules/memory/how_to_guides.html) to see which one fits your use case.

# ### Chat Message History

# In[61]:


from langchain.memory import ChatMessageHistory
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)

history = ChatMessageHistory()

history.add_ai_message("hi!")

history.add_user_message("what is the capital of france?")


# In[62]:


history.messages


# In[63]:


ai_response = chat(history.messages)
ai_response


# In[64]:


history.add_ai_message(ai_response.content)
history.messages


# ## Chains ⛓️⛓️⛓️
# Combining different LLM calls and action automatically
# 
# Ex: Summary #1, Summary #2, Summary #3 > Final Summary
# 
# Check out [this video](https://www.youtube.com/watch?v=f9_BWhCI4Zo&t=2s) explaining different summarization chain types
# 
# There are [many applications of chains](https://python.langchain.com/en/latest/modules/chains/how_to_guides.html) search to see which are best for your use case.
# 
# We'll cover two of them:

# ### 1. Simple Sequential Chains
# 
# Easy chains where you can use the output of an LLM as an input into another. Good for breaking up tasks (and keeping your LLM focused)

# In[65]:


from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain

llm = OpenAI(temperature=1, openai_api_key=openai_api_key)


# In[66]:


template = """Your job is to come up with a classic dish from the area that the users suggests.
% USER LOCATION
{user_location}

YOUR RESPONSE:
"""
prompt_template = PromptTemplate(input_variables=["user_location"], template=template)

# Holds my 'location' chain
location_chain = LLMChain(llm=llm, prompt=prompt_template)


# In[67]:


template = """Given a meal, give a short and simple recipe on how to make that dish at home.
% MEAL
{user_meal}

YOUR RESPONSE:
"""
prompt_template = PromptTemplate(input_variables=["user_meal"], template=template)

# Holds my 'meal' chain
meal_chain = LLMChain(llm=llm, prompt=prompt_template)


# In[68]:


overall_chain = SimpleSequentialChain(chains=[location_chain, meal_chain], verbose=True)


# In[69]:


review = overall_chain.run("Rome")


# ### 2. Summarization Chain
# 
# Easily run through long numerous documents and get a summary. Check out [this video](https://www.youtube.com/watch?v=f9_BWhCI4Zo) for other chain types besides map-reduce

# In[70]:


from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = TextLoader('data/PaulGrahamEssays/disc.txt')
documents = loader.load()

# Get your splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)

# Split your docs into texts
texts = text_splitter.split_documents(documents)

# There is a lot of complexity hidden in this one line. I encourage you to check out the video above for more detail
chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
chain.run(texts)


# ## Agents 🤖🤖
# 
# Official LangChain Documentation describes agents perfectly (emphasis mine):
# > Some applications will require not just a predetermined chain of calls to LLMs/other tools, but potentially an **unknown chain** that depends on the user's input. In these types of chains, there is a “agent” which has access to a suite of tools. Depending on the user input, the agent can then **decide which, if any, of these tools to call**.
# 
# 
# Basically you use the LLM not just for text output, but also for decision making. The coolness and power of this functionality can't be overstated enough.
# 
# Sam Altman emphasizes that the LLMs are good '[reasoning engine](https://www.youtube.com/watch?v=L_Guz73e6fw&t=867s)'. Agent take advantage of this.

# ### Agents
# 
# The language model that drives decision making.
# 
# More specifically, an agent takes in an input and returns a response corresponding to an action to take along with an action input. You can see different types of agents (which are better for different use cases) [here](https://python.langchain.com/en/latest/modules/agents/agents/agent_types.html).

# ### Tools
# 
# A 'capability' of an agent. This is an abstraction on top of a function that makes it easy for LLMs (and agents) to interact with it. Ex: Google search.
# 
# This area shares commonalities with [OpenAI plugins](https://platform.openai.com/docs/plugins/introduction).

# ### Toolkit
# 
# Groups of tools that your agent can select from
# 
# Let's bring them all together:

# In[71]:


from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
import json

llm = OpenAI(temperature=0, openai_api_key=openai_api_key)


# In[72]:


serpapi_api_key=os.getenv("SERP_API_KEY", "YourAPIKey")


# In[73]:


toolkit = load_tools(["serpapi"], llm=llm, serpapi_api_key=serpapi_api_key)


# In[74]:


agent = initialize_agent(toolkit, llm, agent="zero-shot-react-description", verbose=True, return_intermediate_steps=True)


# In[75]:


response = agent({"input":"what was the first album of the" 
                    "band that Natalie Bergman is a part of?"})


# ![Wild Belle](data/WildBelle1.png)

# 🎵Enjoy🎵
# https://open.spotify.com/track/1eREJIBdqeCcqNCB1pbz7w?si=c014293b63c7478c
