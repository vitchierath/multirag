import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import requests
import re
import json

# Load environment variables
load_dotenv()

# Fetch OpenRouter API key
api_key = os.getenv("OPENROUTER_API_KEY")

# Show only first few characters for security
if not api_key:
    st.error("No API key found. Please check your .env file.")
else:
    st.write(f"API Key loaded: {api_key[:5]}...")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Prompt Template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Use the following context to answer the question concisely. "
        "If the context mentions founders, list their names explicitly. "
        "If no founder information is available, say so:\n"
        "{context}\n\nQuestion: {question}\nAnswer:"
    )
)

# Retrieval function
def retrieve_chunks(query, k=3):
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(query, k=k)
    return docs

# RAG Answer Generation using OpenRouter
def generate_answer(query):
    retrieved_docs = retrieve_chunks(query, k=3)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    prompt = prompt_template.format(context=context, question=query)

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://yourdomain.com",  # Replace with your domain or "localhost"
                "X-Title": "RAG Assistant"
            },
            json={
                "model": "mistralai/mixtral-8x7b-instruct",
                "messages": [{"role": "user", "content": prompt}]
            }
        )

        if response.status_code != 200:
            raise Exception(response.json())

        data = response.json()
        return {
            "answer": data["choices"][0]["message"]["content"],
            "context": context,
            "retrieved_chunks": retrieved_docs
        }

    except Exception as e:
        st.error(f"Error calling OpenRouter: {str(e)}")
        return {
            "answer": f"Error: {str(e)}",
            "context": context,
            "retrieved_chunks": retrieved_docs
        }

# Calculator tool
def calculator(query):
    try:
        expression = re.search(r"calculate\s+(.+)", query, re.IGNORECASE).group(1)
        result = eval(expression, {"__builtins__": {}})
        return f"Result: {result}"
    except:
        return "Error: Invalid calculation."

# Dictionary tool
def define_word(query):
    try:
        word = re.search(r"define\s+(\w+)", query, re.IGNORECASE).group(1)
        response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}")
        if response.status_code == 200:
            data = response.json()
            meaning = data[0]["meanings"][0]["definitions"][0]["definition"]
            return f"Definition of {word}: {meaning}"
        return f"No definition found for {word}."
    except:
        return "Error: Invalid word or API issue."

# Query Router
def agent_workflow(query):
    st.write(f"Processing query: {query}")
    if "calculate" in query.lower():
        st.write("Routing to calculator tool")
        return {
            "tool": "calculator",
            "answer": calculator(query),
            "context": None,
            "retrieved_chunks": []
        }
    elif "define" in query.lower():
        st.write("Routing to dictionary tool")
        return {
            "tool": "dictionary",
            "answer": define_word(query),
            "context": None,
            "retrieved_chunks": []
        }
    else:
        st.write("Routing to RAG pipeline")
        result = generate_answer(query)
        return {
            "tool": "RAG",
            "answer": result["answer"],
            "context": result["context"],
            "retrieved_chunks": result["retrieved_chunks"]
        }

# Streamlit UI
st.title("RAG-Powered Q&A Assistant")
query = st.text_input("Enter your question:")

if query:
    result = agent_workflow(query)
    st.write(f"**Tool used**: {result['tool']}")
    st.write(f"**Answer**: {result['answer']}")
    if result["context"]:
        st.write("**Retrieved Context**:")
        for i, doc in enumerate(result["retrieved_chunks"], 1):
            st.write(f"Chunk {i}: {doc.page_content}")
