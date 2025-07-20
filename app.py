import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- Page Configuration ---
st.set_page_config(
    page_title="Income Prediction App 💰",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Data Loading and Caching ---
@st.cache_data
def load_data(file_path):
    """Loads, cleans, and prepares the UCI Adult dataset."""
    df = pd.read_csv(file_path)
    # Clean column names and string data
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()

    # Remove the 'fnlwgt' column as it's not a predictive feature
    df.drop('fnlwgt', axis=1, inplace=True, errors='ignore')

    # Replace '?' with NaN and drop rows with any missing values
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)

    # Convert target variable 'income' to binary
    df['income'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
    return df

# --- Model Training and Caching ---
@st.cache_resource
def train_model(df):
    """Preprocesses data and trains a Logistic Regression model."""
    X = df.drop('income', axis=1)
    y = df['income']

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    # Create a robust preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Create the full model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    # Split data and train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)

    return model_pipeline, X_test, y_test

# --- Main Application UI ---
st.title("Income Level Prediction App 💰")
st.write(
    "This app predicts whether an individual's annual income is above or below $50K "
    "based on their demographic and employment data."
)
st.write("---")

# Load data and train the model
try:
    df = load_data('adult 3.csv')
    model, X_test, y_test = train_model(df)
except FileNotFoundError:
    st.error("Error: `adult 3.csv` not found. Please place the dataset in the same directory.")
    st.stop()

# --- Sidebar for User Input ---
st.sidebar.header("👤 User Input Features")

def user_input_features(data):
    """Creates sidebar widgets for user input and returns a DataFrame."""
    inputs = {}
    for col in data.drop('income', axis=1).columns:
        if data[col].dtype == 'object':
            options = sorted(data[col].unique())
            inputs[col] = st.sidebar.selectbox(f"Select {col.replace('-', ' ').title()}", options)
        else:
            min_val, max_val = int(data[col].min()), int(data[col].max())
            default_val = int(data[col].median()) # Use median for a more robust default
            inputs[col] = st.sidebar.slider(f"Select {col.replace('-', ' ').title()}", min_val, max_val, default_val)
    return pd.DataFrame([inputs])

input_df = user_input_features(df)

# --- Prediction and Output ---
st.header("🔮 Prediction")
col1, col2 = st.columns([1, 2])

with col1:
    st.write("**Your Selections:**")
    st.dataframe(input_df.T.rename(columns={0: 'Value'}), use_container_width=True)

with col2:
    if st.button("Predict Income", type="primary", use_container_width=True):
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        st.subheader("Prediction Result")
        income_level = ">$50K" if prediction == 1 else "<=$50K"
        confidence = prediction_proba[prediction]

        if income_level == ">$50K":
            st.success(f"**Predicted Income:** {income_level}", icon="🎉")
        else:
            st.info(f"**Predicted Income:** {income_level}", icon="💵")

        st.metric(label="**Confidence Score**", value=f"{confidence:.2%}")

        st.write("**Prediction Probabilities:**")
        st.progress(prediction_proba[1], text=f"Probability of earning >$50K")

st.write("---")

# --- Model Performance Metrics ---
with st.expander("📊 See Model Performance Metrics"):
    st.write(
        "These metrics evaluate the model's performance on a separate test set "
        "that it did not see during training."
    )
    y_pred = model.predict(X_test)

    # Calculate and display metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
    m2.metric("Precision", f"{precision_score(y_test, y_pred):.2%}")
    m3.metric("Recall", f"{recall_score(y_test, y_pred):.2%}")
    m4.metric("F1-Score", f"{f1_score(y_test, y_pred):.2%}")

# --- Data Explorer ---
with st.expander("📂 Explore the Training Dataset"):
    st.write("View, sort, and filter the raw data used for training.")
    st.dataframe(df, use_container_width=True)