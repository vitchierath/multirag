import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import requests
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Set OpenRouter API Key (Required Format: Bearer <key>)
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.error("No OpenRouter API key found. Please check your .env file.")
    st.stop()
os.environ["OPENAI_API_KEY"] = f"Bearer {api_key}"

# Display partial key to confirm load
st.success(f"API Key loaded: {api_key[:10]}...")

# Initialize components
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load FAISS vector store
def retrieve_chunks(query, k=3):
    try:
        vector_store = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        docs = vector_store.similarity_search(query, k=k)
        return docs
    except Exception as e:
        st.error(f"Failed to load FAISS index: {e}")
        return []

# Initialize OpenRouter chat model
llm = ChatOpenAI(
    model="mistralai/mixtral-8x7b-instruct",
    openai_api_base="https://openrouter.ai/api/v1"
)

# Prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""Use the following context to answer the question concisely.
If the context mentions founders, list their names explicitly.
If no founder information is available, say so.

{context}

Question: {question}
Answer:"""
)

# Answer generation function
def generate_answer(query):
    retrieved_docs = retrieve_chunks(query, k=3)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    prompt = prompt_template.format(context=context, question=query)
    try:
        response = llm.invoke(prompt)
        return {
            "tool": "RAG",
            "answer": response.content,
            "context": context,
            "retrieved_chunks": retrieved_docs
        }
    except Exception as e:
        return {
            "tool": "RAG",
            "answer": f"Error calling OpenRouter: {str(e)}",
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

# Agent workflow
def agent_workflow(query):
    if "calculate" in query.lower():
        return {
            "tool": "calculator",
            "answer": calculator(query),
            "context": None,
            "retrieved_chunks": []
        }
    elif "define" in query.lower():
        return {
            "tool": "dictionary",
            "answer": define_word(query),
            "context": None,
            "retrieved_chunks": []
        }
    else:
        return generate_answer(query)

# Streamlit UI
st.title("🔍 RAG-Powered Q&A Assistant")

query = st.text_input("Enter your question:")
if query:
    result = agent_workflow(query)
    st.markdown(f"**Tool used**: `{result['tool']}`")
    st.markdown(f"**Answer**: {result['answer']}")
    if result["context"]:
        st.markdown("**Retrieved Context:**")
        for i, doc in enumerate(result["retrieved_chunks"], 1):
            st.markdown(f"**Chunk {i}:** {doc.page_content}")
