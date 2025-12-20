#streamlit: Used for developing the web application interface.
import streamlit as st

## Pdf reader
from langchain_community.document_loaders import PyPDFLoader


# documents are then split into manageable chunks using the RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

#OllamaEmbeddings:Generates vector embeddings from documents, which are then used for similarity searches in the FAISS vector store.
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
#A prompt template for chat models
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains import create_retrieval_chain
#FAISS (Facebook AI Similarity Search): Utilized as the vector store for efficient similarity search and retrieval.
from langchain_community.vectorstores import FAISS 

#LLM Model Ollama's LLaMA 3.1.
# from langchain_community.llms import Ollama

#Create a chain for passing a list of Documents to a model. 
from langchain.chains.combine_documents import create_stuff_documents_chain

#dotenv: Manages environment variables securely.
from dotenv import load_dotenv   
import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI


# Loads all the required API Keys
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] ="true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT="stock_market"






if "vector" not in st.session_state:   

    # step 1: The process begins by loading smm documents using PyPDFLoader.
    st.session_state.loader=PyPDFLoader('ssm.pdf') 
    st.session_state.docs=st.session_state.loader.load()

    # step 2: These documents are then split into manageable chunks using the RecursiveCharacterTextSplitter.
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)  
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])

    #step 3: The document chunks are transformed into high-dimensional vectors using GoogleGenerativeAIEmbeddings.
    st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    #step 4: The embeddings are stored in FAISS, which serves as the vector store.
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)


# step 5
#Users interact with the application through a Streamlit interface, where they input their queries.

# The retrieval chain is initiated, combining the retrieved documents with the ChatPromptTemplate
#  to generate a contextually accurate response using Ollama's LLaMA 3.1.

retriever=st.session_state.vectors.as_retriever()


# Design ChatPrompt Template
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
<context>
{context}
</context>
Question: {input}""")


## LLM Model LLama3.1
llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.3)

# Combine chat prompt template and Ollama llamma3.1 llm model 
document_chain=create_stuff_documents_chain(llm,prompt)

# combine retriver vector store relevent info with document chain
retrieval_chain=create_retrieval_chain(retriever,document_chain)


# Streamlit Application

st.title('StockMarket tutorial with Mishti 💁')

prompt=st.text_input("Ask your question about Stock Market and SSM")

if prompt:
    start=time.process_time()
    response=retrieval_chain.invoke({"input":prompt})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
    

