# Stock Market Analysis Application using LLM.


![image](https://github.com/user-attachments/assets/369ac1b6-7b2c-43bd-a4fe-73091b3c4bc1)

# 🗒️ StockMarket Tutorial with B 
![Screenshot 2024-08-15 131929](https://github.com/user-attachments/assets/a5e9f8bc-e99a-4e50-876e-cf4dd20a3433)
![Screenshot 2024-08-15 131950](https://github.com/user-attachments/assets/1dd3eaf4-2cc0-4c00-b582-6b7fcd1358e6)



📈**Introduction**
This question answer tab is designed to interactively answer stock market-related questions using content extracted from a PDF document. Built using Streamlit, LangChain, and Ollama LLaMA3.1, the app processes PDF documents, splits them into manageable chunks, and uses a vector database for document retrieval and question answering.

**Features**
Load and process PDF documents for stock market tutorials.
Answer questions interactively based on the content of the PDFs.
Perform document similarity searches to find relevant information.

**Usage**
To run the application:

**Load PDF Documents**

The application currently processes a PDF document named ssm.pdf located in the project directory. You can replace this file with your own PDF document.

**Usage**
To run the application:

**Load PDF Documents**

The application currently processes a PDF document named ssm.pdf located in the project directory. You can replace this file with your own PDF document.
streamlit run app.py
Interact with the App

Open your web browser and navigate to the local Streamlit server (usually http://localhost:8501).
Input your question related to the stock market in the text box.
The app will retrieve and display the most relevant content from the PDF document, along with the answer.

**Code Walkthrough**
The following steps describe how the application works:

**Loading and Splitting the PDF Document**
The application uses PyPDFLoader to load the PDF document and RecursiveCharacterTextSplitter to split the content into chunks.

loader = PyPDFLoader('ssm.pdf')
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs[:50])

**Embedding and Vector Store**

The application creates embeddings using the OllamaEmbeddings model and stores them in a FAISS vector database.
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vectors = FAISS.from_documents(documents, embeddings)

**Designing the Chat Prompt Template**

The prompt is designed to instruct the model to answer questions based on the provided context.

prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
Think step by step before providing a detailed answer.
<context>
{context}
</context>
Question: {input}""")


**Setting Up the Retrieval Chain**

A retrieval chain is created to retrieve the relevant documents and generate answers using the Ollama LLaMA3.1 model.
retrieval_chain = create_retrieval_chain(retriever, document_chain)

Handling User Input

The user can input questions through the Streamlit interface, and the application will display the response and relevant document chunks.
prompt = st.text_input("Input your prompt here")
if prompt:
    response = retrieval_chain.invoke({"input": prompt})
    st.write(response['answer'])

**Conclusion**
This application offers an interactive way to explore and learn about the stock market using PDF documents. By leveraging advanced AI models and vector search techniques, it provides accurate and contextually relevant answers to user queries.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 🛠️ Chatbot with Tools

![Screenshot 2024-08-15 122611](https://github.com/user-attachments/assets/1a05ff3b-f05c-4242-858b-666d1e4b7751)


This chatbot tab demonstrates how to build a chatbot using Streamlit, LangChain, and the Ollama LLaMA3.1 model. The chatbot can perform basic mathematical operations such as addition and multiplication, and also respond to natural language inputs. This application leverages tools in LangChain to extend the chatbot's functionality.

**📋 Features**


Math Operations: The chatbot can add or multiply two integers.
Natural Language Processing: The chatbot can respond to natural language inputs by utilizing the Ollama LLaMA3.1 model.
Tool Integration: The chatbot can select and execute tools based on user input using a JSON-based command structure.

**🚀 Getting Started**


Prerequisites
Python 3.8 or higher
Streamlit
LangChain
Ollama LLaMA3.1 model

https://github.com/bhagyeshmawale/GenAi_MultiPerpose_Application.git


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

# Chat with PDF using B💁
![Screenshot 2024-08-15 160119](https://github.com/user-attachments/assets/908f7600-10b4-48a7-91f9-27caa93734c5)


📚 **Chat with PDFs using Streamlit and Google Generative AI**
This summarization tab allows you to upload and query multiple PDF documents using a Streamlit application. The app leverages Google Generative AI (Gemini) and FAISS for embedding and similarity search, allowing for interactive, conversational responses based on the content of the PDFs.

🛠️ **Features**
**Upload Multiple PDFs:** Users can upload multiple PDF files for processing.
Extract Text: Text is extracted from the uploaded PDFs.
Text Chunking: The extracted text is split into chunks to optimize embedding and search processes.
Vector Store Creation: A FAISS vector store is created from the text chunks for efficient querying.
Interactive QA: Users can input questions, and the app provides detailed answers based on the PDF content.
Downloadable Responses: Users can download the generated answers as a text file.
🧰 **Installation**
Clone the Repository:

bash
Copy code
git clone <repository-url>
cd <repository-directory>
Install Dependencies:
Make sure you have Python installed, then run:

bash
Copy code
pip install -r requirements.txt
Set Up Environment Variables:
Create a .env file in the project root directory and add your Google API key:

makefile
Copy code
GOOGLE_API_KEY=your_google_api_key_here
Run the Application:

bash
Copy code
streamlit run app.py


**📝 How It Works**
**PDF Upload and Processing:**

The application allows users to upload multiple PDF files via the sidebar.
On clicking "Submit & Process," the application reads the PDFs and extracts their text.
Text Chunking:

The extracted text is split into manageable chunks using the RecursiveCharacterTextSplitter.

**Vector Store Creation:**
The text chunks are embedded using Google Generative AI embeddings and stored in a FAISS vector store.

**Question Answering:**

Users can input questions related to the content of the PDFs.
The system retrieves relevant chunks using similarity search and generates a detailed response.
Download Response:

The generated response is displayed on the page, and users can download it as a text file.

**⚙️ Code Structure**
get_pdf_text(pdf_docs): Extracts text from the provided PDF documents.

get_text_chunks(text): Splits the extracted text into chunks for better processing.

get_vector_store(text_chunks): Creates and saves a FAISS vector store from the text chunks.

get_conversational_chain(): Sets up the QA model using Google Generative AI and prepares it for generating responses.

user_input(user_question): Handles user queries, retrieves relevant text chunks, and generates a response.

pdf_summary(): Manages the main Streamlit UI and orchestrates the PDF processing and user interactions.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Data Analysis and Machine Learning Models

**Upload Data**

![image](https://github.com/user-attachments/assets/9ec2c4f6-2c9f-4d54-93a3-b465474a7330)


**EDA**

![Screenshot 2024-08-15 160722](https://github.com/user-attachments/assets/961c6124-6c53-4e84-b932-4a4899663a32)
![Screenshot 2024-08-15 160743](https://github.com/user-attachments/assets/5e87a350-d041-4001-adac-e431f1714611)

**Feature Selection**

![Screenshot 2024-08-15 160841](https://github.com/user-attachments/assets/65f70a86-27d4-4aeb-9111-287d0116582e)

**Model Selection | Train & Test variables | Model Validation | Accuracy**

![Screenshot 2024-08-15 161621](https://github.com/user-attachments/assets/3d77a4b8-07aa-4f3b-ba88-2ce66ca31199)

This project provides a Streamlit web application that allows users to upload datasets, perform Exploratory Data Analysis (EDA), select features, and train machine learning models. The application supports CSV, Excel, and JSON file formats.

🚀 Features
**Upload CSV/Excel/JSON File:**

Users can upload a dataset in CSV, Excel (.xlsx, .xls), or JSON format.
The uploaded file is processed and stored for further analysis.
**Exploratory Data Analysis (EDA):**

Generate summary statistics for the dataset.
Visualize the correlation matrix using an interactive heatmap.
**Feature Selection:**

Select specific columns from the dataset to be used as features.
Store the selected features for model training.
**Model Selection and Training:**

Choose between Regression and Classification models.
Select specific models such as Linear Regression, Random Forest Regressor, Logistic Regression, and Random Forest Classifier.
Perform train-test split and train the selected model.
Display model accuracy after training.

**📄 Code Structure**


**upload_page():** Handles the file upload functionality and stores the uploaded data in the session state.


**eda_page():** Performs exploratory data analysis, including summary statistics and correlation matrix visualization.


**feature_selection_page():** Allows users to select features from the dataset for model training.


**model_selection_page():** Facilitates model selection, train-test split, model training, and displays accuracy metrics.

**🖥️ Usage**

**Upload File:** Start by uploading a CSV, Excel, or JSON file using the file uploader.


**Perform EDA:** Navigate to the EDA page to explore summary statistics and the correlation matrix.

**Select Features:** Use the feature selection page to choose the columns to be used in model training.

**Train Model:** Select the type of model (Regression or Classification), choose the specific model, and train it using the selected features.

**📦 Dependencies**

**Streamlit:** Web framework for interactive applications.

**Pandas:** Data manipulation and analysis library.

**Scikit-learn:** Machine learning library for model training.

**Plotly:** Visualization library for interactive charts.

















