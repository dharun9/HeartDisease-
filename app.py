import streamlit as st
import pickle

st.title("Heart Disease Recognition")

# Add input fields for the specified features
sex = st.number_input("Insert the Sex (0 for female, 1 for male)")
cp = st.number_input("Insert the Chest Pain Type")
trestbps = st.number_input("Insert the Resting Blood Pressure")
chol = st.number_input("Insert the Cholesterol Level")
fbs = st.number_input("Insert the Fasting Blood Sugar (0 for False, 1 for True)")
restecg = st.number_input("Insert the Resting ECG")
thalach = st.number_input("Insert the Max Heart Rate")
exang = st.number_input("Insert Exercise-Induced Angina (0 for No, 1 for Yes)")
oldpeak = st.number_input("Insert the Oldpeak")
slope = st.number_input("Insert the Slope")
ca = st.number_input("Insert the Number of Major Vessels (0-3)")
thal = st.number_input("Insert the Thalassemia Type")

# Provide a default value for the missing feature
missing_feature = 0  # You can change this to an appropriate default value

submit = st.button("Predict")
if submit:
    with open("knn_model.pkl", "rb") as f:
        knn_model = pickle.load(f)

    # Prepare the input data with the missing feature
    input_data = [[sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, missing_feature]]

    # Predict heart disease
    output = knn_model.predict(input_data)

    st.success("Predicted Heart Disease (0 for No, 1 for Yes): " + str(output[0]))
