import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set page title
st.title("Heart Disease Prediction")

# Load the trained KNN model
try:
    with open("knn_model.pkl", "rb") as f:
        knn_model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file 'knn_model.pkl' not found. Please make sure it exists in the correct location.")
except Exception as e:
    st.error(f"An error occurred while loading the model: {str(e)}")

# Function to make predictions
def predict_heart_disease(sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, missing_feature):
    global knn_model
    # Prepare the input data
    input_data = np.array([[sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, missing_feature]])
    
    # Predict heart disease
    prediction = knn_model.predict(input_data)
    
    return prediction[0]

# Add input fields for user input
sex = st.number_input("Sex (0 for female, 1 for male)", min_value=0, max_value=1)
cp = st.number_input("Chest Pain Type", min_value=0)
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol Level")
fbs = st.number_input("Fasting Blood Sugar (0 for False, 1 for True)", min_value=0, max_value=1)
restecg = st.number_input("Resting ECG")
thalach = st.number_input("Max Heart Rate")
exang = st.number_input("Exercise-Induced Angina (0 for No, 1 for Yes)", min_value=0, max_value=1)
oldpeak = st.number_input("Oldpeak")
slope = st.number_input("Slope")
ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3)
thal = st.number_input("Thalassemia Type")

# Default value for the missing feature
missing_feature = st.number_input("Missing Feature (e.g., 0)", min_value=0)

# Make prediction when the user clicks the button
if st.button("Predict Heart Disease"):
    # Predict using the model
    prediction = predict_heart_disease(sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, missing_feature)

    # Display the prediction result
    if prediction == 1:
        st.error("Prediction: Heart Disease (1)")
    else:
        st.success("Prediction: No Heart Disease (0)")

    # Create a bar chart to visualize input values
    input_values = [sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, missing_feature]
    input_labels = ["Sex", "Chest Pain", "Resting BP", "Cholesterol", "Fasting BS", "Resting ECG", "Max HR", "Exang", "Oldpeak", "Slope", "Major Vessels", "Thalassemia", "Missing Feature"]
    
    st.header("Input Values Visualization")
    plt.figure(figsize=(10, 6))
    plt.barh(input_labels, input_values, color='skyblue')
    plt.xlabel("Value")
    plt.title("Input Values")
    st.pyplot()
