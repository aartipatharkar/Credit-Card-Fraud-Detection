import streamlit as st

from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.train_model import train_model
from src.evaluate_model import evaluate_model
from src.predict import predict_fraud

st.title("Financial Fraud Detection System")

# Load dataset
df = load_data()

st.write("Dataset Columns:")
st.write(df.columns)

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Preprocess
df = preprocess_data(df)

# Train model with progress message
with st.spinner("Training model... Please wait"):

    model, features, target = train_model(df)

    X = df[features]
    y = df[target]

    accuracy = evaluate_model(
        model,
        X,
        y
    )

st.success("Model training completed")

st.subheader("Model Performance")

st.success(
    f"Model Accuracy: {round(accuracy, 4)}"
)

st.write("Features Used:")
st.write(features[:5])

# Prediction section
st.subheader("Check Transaction")

input_values = []

for feature in features[:5]:

    value = st.number_input(
        f"Enter {feature}",
        value=0.0
    )

    input_values.append(value)

if st.button("Detect Fraud"):

    result = predict_fraud(input_values)

    if result == 1:

        st.error("Fraud Detected")

    else:

        st.success("Transaction Safe")
