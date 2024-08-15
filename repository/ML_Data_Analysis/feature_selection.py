import streamlit as st
import pandas as pd


def feature_selection_page():
    if 'dataframe' in st.session_state:
        df = st.session_state.dataframe
        st.title("Feature Selection")
        
        # Create checkboxes for each column
        columns = df.columns
        selected_columns = []
        for column in columns:
            if st.checkbox(f"Select {column}", value=True):
                selected_columns.append(column)
        
        # Store selected columns in session state
        st.session_state.selected_columns = selected_columns
        
        # Display selected columns
        st.write("Selected columns:", selected_columns)
        
        # # Pass selected columns to model function
        # if st.button("Run Model"):
        #     if 'model_function' in st.session_state:
        #         model_function = st.session_state.model_function
        #         result = model_function(df[selected_columns])
        #         st.write(result)
        #     else:
        #         st.warning("Model function is not defined.")
    else:
        st.warning("Please upload a CSV file first.")

feature_selection_page()