import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def model_selection_page():
    st.title("Model Selection and Training")

    # Step 1: Select Model Type
    model_type = st.selectbox("Select Model Type", ["Regression", "Classification"])

    # Step 2: Select Specific Model Based on Model Type
    if model_type == "Regression":
        model_name = st.selectbox("Select Regression Model", ["Linear Regression", "Random Forest Regressor"])
    elif model_type == "Classification":
        model_name = st.selectbox("Select Classification Model", ["Logistic Regression", "Random Forest Classifier"])

    # Step 3: Select Features for Training and Testing
    if 'dataframe' in st.session_state:
        df = st.session_state.dataframe
        features = df.columns.tolist()
        train_features = st.multiselect("Select Training Features", features)
        target_feature = st.selectbox("Select Target Feature", features)

        # Step 4: Train-Test Split
        X = df[train_features]
        y = df[target_feature]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 5: Initialize and Train Model
        if st.button("Train Model"):
            model = None
            if model_name == "Linear Regression":
                model = LinearRegression()
            elif model_name == "Random Forest Regressor":
                model = RandomForestRegressor()
            elif model_name == "Logistic Regression":
                model = LogisticRegression()
            elif model_name == "Random Forest Classifier":
                model = RandomForestClassifier()

            # Fit the model
            model.fit(X_train, y_train)
            st.success(f"{model_name} trained successfully!")

            # Optionally, display the model's performance
            accuracy = model.score(X_test, y_test)
            st.write(f"Model Accuracy: {accuracy:.2f}")

    else:
        st.warning("Please upload a dataset first.")

# You can call this function within your Streamlit app to render the model selection and training page
model_selection_page()
