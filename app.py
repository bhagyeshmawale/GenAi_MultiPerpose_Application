import streamlit as st
from PIL import Image

image = Image.open('static/genai1.jpg')

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    if st.button("Log in"):
        st.session_state.logged_in = True
        st.rerun()
    st.image(image, caption='Stock Market Analysis Application using LLM', use_column_width=True)

def logout():
    if st.button("Log out"):
        st.session_state.logged_in = False
        st.rerun()
    st.image(image, caption='Stock Market Analysis Application using LLM', use_column_width=True)

login_page = st.Page(login, title="Log in", icon=":material/login:")
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

chain_of_thoughts = st.Page(
    "repository/playground/chainOfThoughtReasoning.py", title="Reasoning", icon=":material/dashboard:"
)
chatbot = st.Page("repository/playground/chatbot.py", title="Chatbot", icon=":material/robot:")
Question_Answers = st.Page(
    "repository/playground/questionAnswer.py", title="Question Answers", icon=":material/question_answer:",default=True
)

pdf_word_text = st.Page("repository/summarization/pdf_word_text.py", title="Summarization", icon=":material/search:")


data = st.Page("repository/ML_Data_Analysis/data.py", title="Data", icon=":material/database:")
eda = st.Page("repository/ML_Data_Analysis/eda.py", title="EDA", icon=":material/bar_chart:")
feature_selection = st.Page("repository/ML_Data_Analysis/feature_selection.py", title="Feature Selection", icon=":material/filter:")
model = st.Page("repository/ML_Data_Analysis/model.py", title="Model", icon=":material/robot:")
test = st.Page("repository/ML_Data_Analysis/test.py", title="Validate", icon=":material/search:")
result_summary = st.Page("repository/ML_Data_Analysis/result_summary.py", title="Model Summary", icon=":material/search:")

if st.session_state.logged_in:
    pg = st.navigation(
        {
            "Account": [logout_page],
            "Playground": [Question_Answers,chatbot,chain_of_thoughts],
            "Summarization": [pdf_word_text],
            "ML Data Analysis": [data,eda,feature_selection,model],

        }
    )
else:
    pg = st.navigation([login_page])

pg.run()




  