import streamlit as st
import pandas as pd

import chardet


def upload_page():
    st.header("📊 EDA and Model Training App")
    st.title("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv', 'xlsx', 'json'])
   


    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        st.session_state.dataframe = df
        st.success("File uploaded successfully!")
        

def dataframe_head():
    if 'dataframe' in st.session_state:
        df = st.session_state.dataframe        
        st.write(df.head(15))       
    else:
        st.warning("Please upload a CSV file first.")

upload_page()
dataframe_head()