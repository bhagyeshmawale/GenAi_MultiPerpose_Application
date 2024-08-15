## Pdf reader
from langchain_community.document_loaders import PyPDFLoader

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from langchain_community.llms import Ollama

from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] ="true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT="stock_market"




# loader=PyPDFLoader('ssm.pdf')
# docs=loader.load()

# text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
# documents=text_splitter.split_documents(docs)

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)


# from langchain_text_splitters import CharacterTextSplitter
# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# db = Chroma.from_documents(documents, embedding_function)

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
# db=FAISS.from_documents(documents[:30],OllamaEmbeddings(model="mxbai-embed-large"))



# Design ChatPrompt Template
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
I will tip you $1000 if the user finds the answer helpful. 
<context>
{context}
</context>
Question: {input}""")



llm=Ollama(model="llama3.1")


document_chain=create_stuff_documents_chain(llm,prompt)


if "vector" not in st.session_state:
    st.session_state.embeddings=OllamaEmbeddings(model="mxbai-embed-large")
    st.session_state.loader=PyPDFLoader('ssm.pdf')
    st.session_state.docs=st.session_state.loader.load()

    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)




retriever=st.session_state.vectors.as_retriever()

retrieval_chain=create_retrieval_chain(retriever,document_chain)


st.title('StockMarket tutorial with B')

prompt=st.text_input("Input you prompt here")

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
    

