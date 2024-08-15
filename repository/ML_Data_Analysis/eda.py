import streamlit as st
import pandas as pd
import plotly
import plotly.express as px

def eda_page():
    if 'dataframe' in st.session_state:
        df = st.session_state.dataframe
        st.title("Exploratory Data Analysis")
        st.write('## Summary Statistics')
        st.write(df.describe())

        st.write('## Correlation Matrix')
        corr = df.corr()
        # plotly.heatmap(corr, annot=True)
        fig = px.imshow(corr, text_auto=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please upload a CSV file first.")


eda_page()