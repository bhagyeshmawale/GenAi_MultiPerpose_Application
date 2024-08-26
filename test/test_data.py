import os 
import pandas as pd 

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI 

from langchain_community.llms import Ollama

from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] ="true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT="stock_market"
llm=Ollama(model="llama3.1")

# Importing the data
df = pd.read_csv('http://www-stat.stanford.edu/~tibs/ElemStatLearn/datasets/SAheart.data') 
# Initializing the agent 
agent = create_pandas_dataframe_agent(Ollama(model="llama3.1"), 
              df, verbose=True) 