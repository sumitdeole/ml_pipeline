import streamlit as st
import joblib
import numpy as np
import os

# Load the trained model
# Get the path to the `model` directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_dir = os.path.join(project_root, 'model')

# Ensure the directory exists
os.makedirs(model_dir, exist_ok=True)

# Define the full path to save the model
model_path = os.path.join(model_dir, 'model.joblib')
model = joblib.load(model_path)

# Title and description
st.title("Iris Classification")
st.write("""
This app predicts the species of Iris flowers based on sepal and petal dimensions.
""")

# Input features
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.35)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

# Predict button
if st.button("Predict"):
    # Prepare feature array
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    species = ["Setosa", "Versicolor", "Virginica"][prediction[0]]
    st.write(f"Predicted Species: **{species}**")
