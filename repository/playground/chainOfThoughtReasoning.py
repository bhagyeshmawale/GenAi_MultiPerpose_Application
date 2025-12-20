import os
import time
from bs4 import BeautifulSoup
import re
import requests
import base64
import json
import yfinance as yf
import langchain
from langchain.agents import Tool, initialize_agent
from langchain_community.chat_models import ChatOpenAI
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
import streamlit as st
import warnings
from dotenv import load_dotenv 
import os
from langchain_groq import ChatGroq
from duckduckgo_search.exceptions import RatelimitException

def zero_shot_prompt_chain():
    load_dotenv()

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    groq_api_key=os.environ['GROQ_API_KEY']
    warnings.filterwarnings("ignore")

    def get_base64(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()


    def set_background(png_file):
        bin_str = get_base64(png_file)
        page_bg_img = '''
        <style>
        .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        }
        </style>
        ''' % bin_str
        st.markdown(page_bg_img, unsafe_allow_html=True)


    # set_background('bcg_light.png')
    st.header('Stock Recommendation System with B')
    #importing api key as environment variable

    # openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
    openai_api_key = os.environ["OPENAI_API_KEY"]


    st.sidebar.write('This tool provides recommendation based on the RAG & ReAct Based Schemes:')
    lst = ['Get Ticker Value',  'Fetch Historic Data on Stock','Get Financial Statements','Scrape the Web for Stock News','LLM ReAct based Verbal Analysis','Output Recommendation: Buy, Sell, or Hold with Justification']

    s = ''

    for i in lst:
        s += "- " + i + "\n"

    st.sidebar.markdown(s)




    if openai_api_key:
        # llm=ChatOpenAI(temperature=0,model_name='gpt-3.5-turbo-0125',openai_api_key=openai_api_key)
        # llm = Ollama(model="llama3.1",temperature=0)
        llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.3)
        # llm=ChatGroq(groq_api_key=groq_api_key,model_name="mixtral-8x7b-32768")

        #Get Historical Stock Closing Price for Last 1 Year
        def get_stock_price(ticker):
            try:
                if "." in ticker:
                    ticker = ticker.split(".")[0]
                ticker = ticker.strip().upper()

                # Primary attempt via Ticker.history
                stock = yf.Ticker(ticker)
                df = stock.history(period="1y", interval="1d")

                # Fallbacks: shorter periods or direct download
                if df is None or df.empty:
                    df = stock.history(period="6mo", interval="1d")
                if df is None or df.empty:
                    df = stock.history(period="3mo", interval="1d")
                if df is None or df.empty:
                    # use yf.download as a more robust fallback
                    try:
                        df = yf.download(tickers=ticker, period="1y", interval="1d", progress=False)
                    except Exception:
                        df = None

                if df is None or df.empty:
                    return f"Unable to retrieve historical data for {ticker}. The ticker may be invalid or the data source is not reachable."

                # Normalize columns and index
                if "Close" in df.columns and "Volume" in df.columns:
                    df = df[["Close", "Volume"]]
                elif "Close" in df.columns:
                    df = df[["Close"]]

                df.index = [str(x).split()[0] for x in list(df.index)]
                df.index.rename("Date", inplace=True)

                # Return last 30 rows for brevity
                return f"Historical data for {ticker}:\n{df.tail(30).to_string()}"
            except Exception as e:
                return f"Error retrieving stock data for {ticker}: {str(e)}"


        #Get News From Web Scraping
        def google_query(search_term):
            if "news" not in search_term:
                search_term = search_term+" stock news"
            url = f"https://www.google.com/search?q={search_term}"
            url = re.sub(r"\s","+",url)
            return url

        #Get Recent Stock News
        def get_recent_stock_news(company_name):
            try:
                # First, try yfinance news via Ticker.news
                try:
                    ticker_guess = None
                    # if user passed a ticker (all caps), use it directly
                    if company_name.isupper() and len(company_name) <= 5:
                        ticker_guess = company_name
                    if ticker_guess is None:
                        # attempt to find ticker from common_tickers mapping
                        for k, v in common_tickers.items():
                            if k in company_name.lower():
                                ticker_guess = v
                                break

                    if ticker_guess:
                        t = yf.Ticker(ticker_guess)
                        try:
                            ynews = t.news
                        except Exception:
                            ynews = None
                        if ynews and isinstance(ynews, list) and len(ynews) > 0:
                            headlines = []
                            for item in ynews[:6]:
                                title = item.get('title') if isinstance(item, dict) else str(item)
                                if title:
                                    headlines.append(title)
                            if headlines:
                                news_string = "\n".join([f"{i+1}. {h}" for i, h in enumerate(headlines)])
                                return f"Recent News for {ticker_guess}:\n\n" + news_string

                except Exception:
                    pass

                # Next, try scraping Google search results (best-effort)
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}
                g_query = google_query(company_name)
                try:
                    res = requests.get(g_query, headers=headers, timeout=10).text
                except Exception:
                    res = ""

                if res:
                    soup = BeautifulSoup(res, "html.parser")
                    news = []
                    for n in soup.find_all("div", "n0jPhd ynAwRc tNxQIb nDgy9d"):
                        if n.text.strip():
                            news.append(n.text.strip())
                    for n in soup.find_all("div", "IJl0Z"):
                        if n.text.strip():
                            news.append(n.text.strip())
                    if len(news) == 0:
                        for n in soup.find_all("h3"):
                            if n.text.strip() and len(n.text) > 10:
                                news.append(n.text.strip())
                                if len(news) >= 4:
                                    break
                    if len(news) > 6:
                        news = news[:4]
                    if len(news) > 0:
                        news_string = "\n".join([f"{i+1}. {n}" for i, n in enumerate(news)])
                        return f"Recent News for {company_name}:\n\n" + news_string

                # Final fallback: Wikipedia summary
                try:
                    wiki_info = wiki.run(company_name)
                    if wiki_info:
                        return f"Company information from Wikipedia:\n{wiki_info[:800]}..."
                except Exception:
                    pass

                return f"No recent news found for {company_name}."
            except Exception as e:
                return f"Error retrieving news for {company_name}: {str(e)}"

        #Get Financial Statements
        def get_financial_statements(ticker):
            try:
                if "." in ticker:
                    ticker = ticker.split(".")[0]
                ticker = ticker.strip().upper()
                company = yf.Ticker(ticker)

                # Try multiple attributes that yfinance provides
                candidates = []
                try:
                    if hasattr(company, 'balance_sheet'):
                        candidates.append(company.balance_sheet)
                except Exception:
                    pass
                try:
                    if hasattr(company, 'quarterly_balance_sheet'):
                        candidates.append(company.quarterly_balance_sheet)
                except Exception:
                    pass
                try:
                    if hasattr(company, 'financials'):
                        candidates.append(company.financials)
                except Exception:
                    pass
                try:
                    if hasattr(company, 'quarterly_financials'):
                        candidates.append(company.quarterly_financials)
                except Exception:
                    pass
                try:
                    if hasattr(company, 'cashflow'):
                        candidates.append(company.cashflow)
                except Exception:
                    pass
                try:
                    if hasattr(company, 'quarterly_cashflow'):
                        candidates.append(company.quarterly_cashflow)
                except Exception:
                    pass

                # Pick the first non-empty dataframe
                data = None
                for item in candidates:
                    if item is None:
                        continue
                    if hasattr(item, 'empty') and not item.empty:
                        data = item
                        break

                # If still no dataframe, try company.info for summary
                if data is None:
                    try:
                        info = company.info
                        if info and isinstance(info, dict) and len(info) > 0:
                            info_snippet = json.dumps({k: info[k] for k in list(info)[:10]}, indent=2)
                            return f"Company info for {ticker}:\n{info_snippet}\n\nNote: Structured financial tables not found via yfinance."
                    except Exception:
                        pass
                    return f"Financial statements not available for {ticker}."

                # Clean and return
                if hasattr(data, 'shape') and data.shape[1] > 6:
                    data = data.iloc[:, :6]
                if hasattr(data, 'dropna'):
                    data = data.dropna(how='all')
                if hasattr(data, 'empty') and data.empty:
                    return f"Financial data for {ticker} is empty."

                return f"Financial Data for {ticker}:\n{data.to_string()}"
            except Exception as e:
                return f"Financial data for {ticker}: Unable to retrieve due to {str(e)}"

        #Initialize Wikipedia Search (no API key needed)
        wiki = WikipediaAPIWrapper()
        
        # Common stock ticker mappings to reduce API calls
        common_tickers = {
            "google": "GOOGL",
            "apple": "AAPL",
            "microsoft": "MSFT",
            "amazon": "AMZN",
            "tesla": "TSLA",
            "nvidia": "NVDA",
            "meta": "META",
            "facebook": "META",
            "alphabet": "GOOGL",
            "netflix": "NFLX",
            "ibm": "IBM",
            "intel": "INTC",
            "amd": "AMD",
            "qualcomm": "QCOM",
            "adobe": "ADBE",
            "nvidia": "NVDA",
            "salesforce": "CRM",
            "oracle": "ORCL",
            "cisco": "CSCO",
            "costco": "COST",
            "jp morgan": "JPM",
            "bank of america": "BAC",
            "goldman sachs": "GS",
            "citigroup": "C",
            "wells fargo": "WFC",
            "berkshire hathaway": "BRK.B",
            "johnson & johnson": "JNJ",
            "pfizer": "PFE",
            "moderna": "MRNA",
            "coca-cola": "KO",
            "pepsi": "PEP",
            "mcdonald's": "MCD",
            "starbucks": "SBUX"
        }
        
        def search_with_wikipedia(query):
            """Wrapper around Wikipedia search for stock ticker lookup"""
            # Check if query contains a common company name first
            query_lower = query.lower()
            for company, ticker in common_tickers.items():
                if company in query_lower:
                    return f"{ticker} is the stock ticker for {company.title()}"
            
            try:
                # Search Wikipedia for the company
                result = wiki.run(query)
                if result and len(result) > 0:
                    # Try to extract ticker from the Wikipedia content
                    if "ticker" in result.lower() or "symbol" in result.lower():
                        return result
                    else:
                        # Return the Wikipedia content with a note
                        return f"Company info from Wikipedia:\n{result}\n\nNote: Could not find explicit ticker symbol. Please provide company name."
                else:
                    return f"No information found for {query} on Wikipedia. Please provide a more specific company name."
            except Exception as e:
                return f"Unable to search for {query}. Error: {str(e)}"
        
        tools = [
        Tool(
            name="Stock Ticker Search",
            func=search_with_wikipedia,
            description="Use to find stock ticker symbols and company information. Searches Wikipedia and a database of common stocks. Input the company name."

        ),
        Tool(
            name = "Get Stock Historical Price",
            func = get_stock_price,
            description="Use when you are asked to evaluate or analyze a stock. This will output historic share price data. You should input the stock ticker to it"

        ),
        Tool(
            name="Get Recent News",
            func= get_recent_stock_news,
            description="Use this to fetch recent news about stocks"
        ),
        Tool(
            name="Get Financial Statements",
            func=get_financial_statements,
            description="Use this to get financial statement of the company. With the help of this data company's historic performance can be evaluated. You should input stock ticker to it"
        )
        ]

        zero_shot_agent=initialize_agent(
            llm=llm,
            agent="zero-shot-react-description",
            tools=tools,
            verbose=True,
            max_iteration=4,
            return_intermediate_steps=False,
            handle_parsing_errors=True
        )

        #Adding predefine evaluation steps in the agent Prompt
        stock_prompt="""You are a financial advisor. Give stock recommendations for given query.
        Everytime first you should identify the company name and get the stock ticker symbol for the stock.
        Answer the following questions as best you can. You have access to the following tools:

        Get Stock Historical Price: Use when you are asked to evaluate or analyze a stock. This will output historic share price data. You should input the stock ticker to it 
        Stock Ticker Search: Use only when you need to get stock ticker from internet, you can also get recent stock related news. Dont use it for any other analysis or task
        Get Recent News: Use this to fetch recent news about stocks
        Get Financial Statements: Use this to get financial statement of the company. With the help of this data company's historic performance can be evaluaated. You should input stock ticker to it

        steps- 
        Note- if you fail in satisfying any of the step below, Just move to next one. If data is unavailable from multiple sources, provide analysis based on the data you DO have.
        1) Get the company name and search for the "company name + stock ticker" on internet. Dont hallucinate extract stock ticker as it is from the text. Output- stock ticker. If stock ticker is not found, stop the process and output this text: This stock does not exist
        2) Use "Get Stock Historical Price" tool to gather stock info. Output- Stock data. If no data available, note it and continue.
        3) Get company's historic financial data using "Get Financial Statements". Output- Financial statement in tabular format. If no data available, note it and continue.
        4) Use this "Get Recent News" tool to search for latest stock related news. Output- Stock news. If no news available, note it and continue.
        5) Based on the data you were able to retrieve, provide analysis for investment choice. If you have limited data, be honest about it and provide analysis on what you DO know. Always try to provide at least a preliminary recommendation even with limited data. Output- Give a single answer if the user should buy, hold or sell. You should Start the answer with Either Buy, Hold, or Sell in Bold after that Justify with available information.

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do, Also try to follow steps mentioned above
        Action: the action to take, should be one of [Get Stock Historical Price, Stock Ticker Search, Get Recent News, Get Financial Statements]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times, if Thought is empty go to the next Thought and skip Action/Action Input and Observation)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
        Begin!

        Question: {input}
        Thought:{agent_scratchpad}"""

        zero_shot_agent.agent.llm_chain.prompt.template=stock_prompt

        if prompt := st.chat_input():
            st.chat_message("user").write(prompt)
            with st.chat_message("assistant"):
                st_callback = StreamlitCallbackHandler(st.container())
                response = zero_shot_agent(f'Is {prompt} a good investment choice right now?', callbacks=[st_callback])
                st.write(response["output"])


zero_shot_prompt_chain()


