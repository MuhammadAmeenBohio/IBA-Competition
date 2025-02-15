import streamlit as st
from pyngrok import ngrok
import time
import os
import signal
import threading
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from dotenv import load_dotenv
from groq import Groq
import transformers

#import torch
#from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the locally stored model and tokenizer
#MODEL_PATH = "./local_model"  # Change to your actual model path
#tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
#model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

# Global variable for last activity time
last_active = time.time()

st.title("RAG-Powered Query App")
st.write("Enter your query below and get responses from the locally stored model.")

# User input
query = st.text_input("Enter your query:")


model_path = "E:\\PRACTICE\\NLP\\Python Code Generator\\all-MiniLM-L6-v2_local"
model_name = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_path)
vector_store = FAISS.load_local("E:\\PRACTICE\\NLP\\Python Code Generator\\faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever(search_kwargs={"k": 15})


retrieved_docs = retriever.get_relevant_documents(query)
context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}\n" + "-"*50 for i, doc in enumerate(retrieved_docs)])

# Initialize Groq client
client = Groq(api_key="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")

# Generate response using Groq
chat_completion = client.chat.completions.create(
    messages=[
        {"role": "system", "content": context},
        {"role": "user", "content": query},
    ],
    model="llama-3.3-70b-versatile",
)

# Print the response
print("\nGenerated Response:")
print(chat_completion.choices[0].message.content)


if query:
    with st.spinner("Processing..."):
        time.sleep(2)  # Simulating processing delay
        response = chat_completion.choices[0].message.content  # Placeholder response
    
    st.success(response)
    st.write("### Context:")
    st.success(context)

# Start ngrok and generate public URL
ngrok_tunnel = None

def start_ngrok():
    global ngrok_tunnel
    ngrok_tunnel = ngrok.connect(8501)
    st.write(f"Public URL: {ngrok_tunnel}")

if st.button("Generate Public URL"):
    start_ngrok()
    st.success("Public URL generated!")


# JavaScript to detect user interaction
st.markdown("""
    <script>
        function keepSessionAlive() {
            fetch("/_stcore/stream");
        }
        setInterval(keepSessionAlive, 5000);
    </script>
""", unsafe_allow_html=True)

# Update the global last_active time when user interacts
last_active = time.time()