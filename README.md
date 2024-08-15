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


