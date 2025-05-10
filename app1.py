import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import requests
import re
from dotenv import load_dotenv
import os
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.base_url = "https://openrouter.ai/api/v1"

# Initialize components
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="Use the following context to answer the question concisely. "
             "If the context mentions founders, list their names explicitly. "
             "If no founder information is available, say so:\n\n"
             "{context}\n\nQuestion: {question}\nAnswer:"
)

# # Call OpenRouter via openai client
# def query_openrouter(prompt):
#     try:
#         response = openai.ChatCompletion.create(
#             model="mistralai/mixtral-8x7b-instruct",
#             messages=[{"role": "user", "content": prompt}]
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         return f"Error calling OpenRouter: {str(e)}

# Function to query OpenRouter API directly
def query_openrouter(prompt):
    try:
        # OpenRouter API call
        response = requests.post(
            "https://openrouter.ai/api/v1/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistralai/mixtral-8x7b-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150  # Adjust based on the length of expected answers
            }
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code} - {response.text}"
    
    except Exception as e:
        return f"Error calling OpenRouter: {str(e)}"


# Retrieve top k similar chunks from FAISS index
def retrieve_chunks(query, k=3):
    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    docs = vector_store.similarity_search(query, k=k)
    return docs

# Generate answer via RAG pipeline
def generate_answer(query):
    retrieved_docs = retrieve_chunks(query, k=3)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    prompt = prompt_template.format(context=context, question=query)
    answer = query_openrouter(prompt)
    return {
        "answer": answer,
        "context": context,
        "retrieved_chunks": retrieved_docs
    }

# Simple calculator tool
def calculator(query):
    try:
        expression = re.search(r"calculate\s+(.+)", query, re.IGNORECASE).group(1)
        result = eval(expression, {"__builtins__": {}})
        return f"Result: {result}"
    except:
        return "Error: Invalid calculation."

# Simple dictionary definition tool
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

# Agent workflow dispatcher
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
