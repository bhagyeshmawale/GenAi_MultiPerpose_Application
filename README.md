Generative AI Multi Purpose Application.
![image](https://github.com/user-attachments/assets/369ac1b6-7b2c-43bd-a4fe-73091b3c4bc1)

![Screenshot 2024-08-15 122611](https://github.com/user-attachments/assets/1a05ff3b-f05c-4242-858b-666d1e4b7751)

please check code in repository/chatbot.py
# 🛠️ Chatbot with Tools
This chatbot tab demonstrates how to build a chatbot using Streamlit, LangChain, and the Ollama LLaMA3.1 model. The chatbot can perform basic mathematical operations such as addition and multiplication, and also respond to natural language inputs. This application leverages tools in LangChain to extend the chatbot's functionality.

📋 Features
Math Operations: The chatbot can add or multiply two integers.
Natural Language Processing: The chatbot can respond to natural language inputs by utilizing the Ollama LLaMA3.1 model.
Tool Integration: The chatbot can select and execute tools based on user input using a JSON-based command structure.

🚀 Getting Started
Prerequisites
Python 3.8 or higher
Streamlit
LangChain
Ollama LLaMA3.1 model

git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.tools import tool
from langchain.tools.render import render_text_description
from langchain_core.output_parsers import JsonOutputParser
from operator import itemgetter

Model Initialization
Initialize the Ollama LLaMA3.1 model with verbosity enabled.

Define Tools
Define the tools available for the chatbot.

Tool Rendering
Render the tools as a text description.

Prompt and Chain Creation
Create a prompt template and link it with the model and output parser.


Customization
You can extend the functionality of this chatbot by defining additional tools and updating the prompt template. Explore the LangChain and Streamlit documentation for more advanced use cases.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
![Screenshot 2024-08-15 131929](https://github.com/user-attachments/assets/a5e9f8bc-e99a-4e50-876e-cf4dd20a3433)
![Screenshot 2024-08-15 131950](https://github.com/user-attachments/assets/1dd3eaf4-2cc0-4c00-b582-6b7fcd1358e6)


🗒️ StockMarket Tutorial with B 
📈Introduction
This application is designed to interactively answer stock market-related questions using content extracted from a PDF document. Built using Streamlit, LangChain, and Ollama LLaMA3.1, the app processes PDF documents, splits them into manageable chunks, and uses a vector database for document retrieval and question answering.

Features
Load and process PDF documents for stock market tutorials.
Answer questions interactively based on the content of the PDFs.
Perform document similarity searches to find relevant information.

Usage
To run the application:

Load PDF Documents

The application currently processes a PDF document named ssm.pdf located in the project directory. You can replace this file with your own PDF document.

Usage
To run the application:

Load PDF Documents

The application currently processes a PDF document named ssm.pdf located in the project directory. You can replace this file with your own PDF document.
streamlit run app.py
Interact with the App

Open your web browser and navigate to the local Streamlit server (usually http://localhost:8501).
Input your question related to the stock market in the text box.
The app will retrieve and display the most relevant content from the PDF document, along with the answer.

Code Walkthrough
The following steps describe how the application works:

Loading and Splitting the PDF Document

The application uses PyPDFLoader to load the PDF document and RecursiveCharacterTextSplitter to split the content into chunks.

loader = PyPDFLoader('ssm.pdf')
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs[:50])

Embedding and Vector Store

The application creates embeddings using the OllamaEmbeddings model and stores them in a FAISS vector database.
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vectors = FAISS.from_documents(documents, embeddings)

Designing the Chat Prompt Template

The prompt is designed to instruct the model to answer questions based on the provided context.

prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
Think step by step before providing a detailed answer.
<context>
{context}
</context>
Question: {input}""")


Setting Up the Retrieval Chain

A retrieval chain is created to retrieve the relevant documents and generate answers using the Ollama LLaMA3.1 model.
retrieval_chain = create_retrieval_chain(retriever, document_chain)

Handling User Input

The user can input questions through the Streamlit interface, and the application will display the response and relevant document chunks.
prompt = st.text_input("Input your prompt here")
if prompt:
    response = retrieval_chain.invoke({"input": prompt})
    st.write(response['answer'])

Conclusion
This application offers an interactive way to explore and learn about the stock market using PDF documents. By leveraging advanced AI models and vector search techniques, it provides accurate and contextually relevant answers to user queries.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



