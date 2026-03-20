
import os
import json
import streamlit as st
from llama_cpp import Llama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None
    Settings = None
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, Docx2txtLoader
)




@st.cache_resource
def load_embedder():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# !/usr/bin/env python
# coding: utf-8

# Now that we've covered the basics of RAG and built an end-to-end real-life application, we are ready to move on to advanced techniques. As we have seen, RAG is a two-part process -- the first is the "retrieval" part which is responsible for fetching the relevant information from a large amount of text.  We achieved this by querying our ChromaDB.  The second is the "generation" part which generates a response based on the query and the retrieved information as input.  We achieved this by constructing a RAG prompt with the queried documents and sending this to an existing model using OpenAI's API.
#
#
# In this lesson, we'll explore a variety of more elaborate systems at the intersection of the vector database and the prompted LLM.
#
# For our text data we will use the textbook *The Mind and Its Education* by George Herbert Betts, which is available on Project Gutenberg here: https://www.gutenberg.org/files/20220/20220-h/20220-h.htm
#
# In this first exercise, we'll get a Chroma vector database and some helper functions intialized so we can rapidly tour a variety of RAG techniques throughout the lesson.

# <details><summary style="display:list-item; font-size:16px; color:blue;">Jupyter Help</summary>
#
# Having trouble testing your work? Double-check that you have followed the steps below to write, run, save, and test your code!
#
# [Click here for a walkthrough GIF of the steps below](https://static-assets.codecademy.com/Courses/ds-python/jupyter-help.gif)
#
# Run all initial cells to import libraries and datasets. Then follow these steps for each question:
#
# 1. Add your solution to the cell with `## YOUR SOLUTION HERE ## `.
# 2. Run the cell by selecting the `Run` button or the `Shift`+`Enter` keys.
# 3. Save your work by selecting the `Save` button, the `command`+`s` keys (Mac), or `control`+`s` keys (Windows).
# 4. Select the `Test Work` button at the bottom left to test your work.
#
# ![Screenshot of the buttons at the top of a Jupyter Notebook. The Run and Save buttons are highlighted](https://static-assets.codecademy.com/Paths/ds-python/jupyter-buttons.png)

# **Setup**
#
# In the following cell, we import Chroma, Langchain, and OpenAI. We also define the metadata of the document we'll be using for RAG.

# In[3]:


import chromadb

book_metadata = {
    "title": "The Mind and Its Education",
    "author": "George Herbert Betts",
    "source_url": "https://www.gutenberg.org/ebooks/20220",
    "filename": "themind.txt"
}
embedding_function = chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
embedding = embedding_function(["Welcome to this RAG course!"])
print(embedding[0][:5]) # we only print the first five values


text_to_embed = "This is my own text to embed."

# Generate embedding of your text
my_embedding = embedding_function([text_to_embed])

# Print the length of the embedding to confirm it has 384 dimensions
print(len(my_embedding[0]))
chroma_client = chromadb.Client()

# Create a new collection
collection = chroma_client.create_collection(name="my_collection")

my_docs = ["This is a document", "This is another document"]
embeddings = embedding_function(my_docs)

# Add some sample data to the collection
collection.add(
    embeddings=embeddings,
    documents=my_docs,
    ids=["id1", "id2"]
)

# Peek at the first few rows of the collection
collection.peek()

my_collection = chroma_client.get_or_create_collection(name="my_collection")

# Generate embeddings for the three documents
docs = ["document one", "document two", "document three"]
embeddings = embedding_function(docs)

# Add the documents and their embeddings to the collection
my_collection.add(
    embeddings=embeddings,
    documents=docs,
    ids=["id1", "id2", "id3"]
)

# Peek at the first few rows of the new collection
my_collection.peek()

cosine_collection = chroma_client.get_or_create_collection(
        name="cosine_collection",
        metadata={"hnsw:space": "cosine"}
    )


cosine_collection.add(
    documents=["In this document, we'll talk all about big cats. Tigers, mountain lions, panthers, and other ferocious felines.",
               "In this document we'll discuss the solar system: moons, planets, and asteroids. We'll also talk about the sun and the stars."],
    ids=["id1", "id2"]
)

# Peek at the first few rows of the cosine_collection
cosine_collection.peek()
# #### Checkpoint 1/3
#
# First, initialize the OpenAI client assigned to `client_openai` by calling `OpenAI()`.
#
# Then, call the client's `.chat.completions.create()` method and pass it the following values:
#
# - for `model`, pass `"gpt-4"`
# - for `messages`, pass a list containing two dicts:
#     - one dict with the `"role"` key `"system"` and the `"content"` key `"You are a helpful assistant connected to a database for document search."`
#     - one dict with the `"role"` key `"user"` and the `"content"` key `prompt`
#
# This `get_completion` helper function will make it easy for us to prompt the model throughout the lesson.

# In[ ]:


# In[4]:


MODEL_PATH = "/home/albasha/models/mistral-7b-instruct-v0.2-gguf/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
CHROMA_PATH = "chroma_db"
OUTPUT_JSON = "output.json"


# ============================
# Load LLM
# ============================
@st.cache_resource
def load_llm():
    try:
        return Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=35,
            verbose=False
        )
    except Exception as e:
        st.error(f"❌ خطأ في تحميل النموذج: {e}")
        st.stop()

def get_completion(prompt):
    response = MODEL_PATH.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system",
             "content": "You're a helpful assistant who retrieves information from external sources and presents them to the user."},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content


# #### Chunking the Document
#
# In the next cell, we'll initialize a recursive text splitter and create chunks of our text with it.
#
# Be sure to execute this cell before moving onto the next checkpoint.

# In[ ]:


# #### Checkpoint 2/3
# Now we will create a new persistent Chroma collection.
#
# Use Chroma's `.PersistentClient()` method to initialize a database that will persist throughout the lesson. Pass the method the desired route to our collection, `"./advanced"`.
#
# Then use the Chroma client's `.get_or_create_collection()` method. Pass the method the `name` `"advanced"` and indicate we'll use cosine similarity with `metadata={"hnsw:space": "cosine"}`.
#
# Fill in the missing sections of the cell.

# In[ ]:


client_chroma = chromadb.PersistentClient("./advanced")
collection = client_chroma.get_or_create_collection(name="advanced", metadata={"hnsw:space": "cosine"})


# #### Uploading the chunks
#
# Now that our collection is initialized we can upload to it the chunks we made earlier.
#
# We enumerate through the list of chunks, accessing the chunk's text in `chunk.page_content`, then add a chunk index to its metadata and upload the document, its id, and its metadata to our Chroma collection.
#
# Be sure to execute this cell before moving on to the next one.

# In[ ]:


# #### Formatting the search results
# Now we'll define a helper function that takes a user query and returns a well-formatted, pseudo-XML string of the search results. This will make it easier to experiment throughout the lesson.
#
# Don't forget to execute this cell before moving on.

# In[ ]:


def populate_rag_query(query, n_results=1):
    search_results = collection.query(query_texts=[query], n_results=n_results)
    result_str = ""
    for idx, result in enumerate(search_results["documents"][0]):
        metadata = search_results["metadatas"][0][idx]
        formatted_result = f"""<SEARCH RESULT>
        <DOCUMENT>{result}</DOCUMENT>
        <METADATA>
        <TITLE>{metadata['title']}</TITLE>
        <AUTHOR>{metadata['author']}</AUTHOR>
        <CHUNK_IDX>{metadata['chunk_idx']}</CHUNK_IDX>
        <URL>{metadata['source_url']}</URL>
        </METADATA>
        </SEARCH RESULT>"""
        result_str += formatted_result
    return result_str


# #### Checkpoint 3/3
#
# Finally, create the RAG prompt to send to the LLM.
#
# Write out the instructions to the model in your own words in the `<INSTRUCTIONS>` section.
#
# Consider how you might guide the model to:
#  - Use the search results effectively
#  - Handle cases where information isn't available
#  - Provide credibility to its answers by citing sources
#
# Note we've included a  `<EXAMPLE CITATION>` that can show the model how its cited sources should look.
#
# To wrap it up, pass the correct variables in the `<USER QUERY>` and `<SEARCH RESULTS>` sections to finish the function.
#

# In[ ]:


def make_rag_prompt(query, result_str):
    return f"""<INSTRUCTIONS>
   Use the search results to answer the user query. Cite the sources in search results with the example citation format provided.
   If an answer is not present in the search results, say so.
   <EXAMPLE CITATION>
   Answer to the user query in your own words, drawn from the search results.
   - "Direct quote from source material backing up the claim" - [Source: Title, Author, Chunk: chunk index, Link: url]
   </EXAMPLE CITATION>
   </INSTRUCTIONS>

    <USER QUERY>
    {query}
    </USER QUERY>

    <SEARCH RESULTS>
    {result_str}
    </SEARCH RESULTS>

    Your answer:"""










chat_memory = [
    {"role": "user", "content": "Tell me about the nature of consciousness."},
    {"role": "assistant", "content": "Consciousness is like a stream that flows from birth to death. It's a continuous process that we can only observe in the present moment. The content of consciousness includes our thoughts, feelings, and will - the processes of knowing, feeling, and choosing."},
    {"role": "user", "content": "That's interesting. How does attention relate to consciousness?"},
    {"role": "assistant", "content": "Attention is the concentration of consciousness on a particular object or thought. It's like a wave in the stream of consciousness, where the object of attention stands out more prominently. Attention allows us to focus our mental energy, making the object of focus clearer and more defined in our mind."},
    {"role": "user", "content": "I've heard attention is important for mental efficiency. Is that true?"},
    {"role": "assistant", "content": "Yes, attention is crucial for mental efficiency. It's like focusing sunlight through a lens - it concentrates mental energy, allowing us to accomplish more in less time. Strong attention skills can significantly enhance our ability to learn, solve problems, and be productive in our mental tasks."}
]

contextual_query = "How does it affect our ability to think and work effectively?"




chat_history = ""

for message in chat_memory:
    role = message["role"]
    content = message["content"]
    chat_history += f"{role}: {content}\n\n"

print("Chat History:")
print(chat_history)


def rewrite_query(query, chat_history):
    prompt = f"""<INSTRUCTIONS>
Given the following chat history and the user's latest query, rewrite the query to include relevant context.
</INSTRUCTIONS>

<CHAT_HISTORY>
{chat_history}
</CHAT_HISTORY>

<LATEST_QUERY>
{query}
</LATEST_QUERY>

Your rewritten query:"""

    return get_completion(prompt)


cqr_query = rewrite_query(contextual_query, chat_history)
print("ORIGINAL QUERY:")
print(contextual_query)
print("CQR QUERY:")
print(cqr_query)


def perform_cqr_rag(query, chat_history, n_results=2):
    # Rewrite the query using the chat history
    refined_query = rewrite_query(query, chat_history)
    result_str = populate_rag_query(refined_query, n_results)
    rag_prompt = make_rag_prompt(refined_query, result_str)
    rag_completion = get_completion(rag_prompt)
    return refined_query, rag_completion

refined_query, rag_completion = perform_cqr_rag(contextual_query, chat_history)
print(rag_completion)




def make_hyde_prompt(query):
    return f"""<INSTRUCTIONS>
Your task is to create a hypothetical document that answers the QUERY shared below.

If you know the answer, then provide it.
If you don't know the answer, try to sound like you do. Use language that would likely be *in* the answer, to the best of your ability.
Use as much language that would likely be in the answer as possible.
Make the answer about a paragraph long.
Output ONLY the hypothetical answer.
</INSTRUCTIONS>

<QUERY>{query}</QUERY>
"""



query = "How do we know what the mind is?"

hyde_query_prompt = make_hyde_prompt(query)
hyde_query = get_completion(hyde_query_prompt)

print(hyde_query)

results = collection.query(
    query_texts=[hyde_query],
    n_results=3
)
print(results)


def answer_query_with_hyde(user_query: str) -> str:
    hyde_prompt = make_hyde_prompt(user_query)
    hyde_query = get_completion(hyde_prompt)
    result_str = populate_rag_query(hyde_query, n_results=3)
    rag_prompt = make_rag_prompt(user_query, result_str)
    rag_completion = get_completion(rag_prompt)

    return rag_completion


user_query = "How do we know what the mind is?"
answer = answer_query_with_hyde(user_query)
print(answer)
