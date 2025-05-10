import streamlit as st
import requests
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# Initialize components
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # From .env
    base_url="https://openrouter.ai/api/v1",
    model="mistralai/mixtral-8x7b-instruct"
)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Use the following context to answer the question concisely. "
        "If the context mentions founders, list their names explicitly. "
        "If no founder information is available, say so:\n\n"
        "{context}\n\nQuestion: {question}\nAnswer:"
    )
)

# Retrieval function
def retrieve_chunks(query, k=3):
    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    docs = vector_store.similarity_search(query, k=k)
    return docs

# Answer generation
def generate_answer(query):
    retrieved_docs = retrieve_chunks(query, k=3)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    prompt = prompt_template.format(context=context, question=query)
    answer = llm.invoke(prompt)
    return {
        "answer": answer.content,
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
