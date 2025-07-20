import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and preprocess objects
model_data = joblib.load('salary_model.joblib')
model = model_data['model']
job_titles = model_data['job_titles']
education_mapping = model_data['education_mapping']
label_encoder = model_data['label_encoder']

st.title("Salary Prediction App")

# Input widgets for user to enter data
age = st.number_input('Age', min_value=18, max_value=70, value=30)
gender = st.selectbox('Gender', label_encoder.classes_)  # Shows original gender labels
education_level = st.selectbox('Education Level', list(education_mapping.keys()))
years_of_exp = st.number_input('Years of Experience', min_value=0, max_value=50, value=5)
job_title = st.selectbox('Job Title', ['Others'] + job_titles)  # Include 'Others' option

# When predict button is clicked
if st.button('Predict Salary'):
    # Prepare the data in the same way as training
    # Gender encoding
    gender_enc = label_encoder.transform([gender])[0]

    # Education mapping
    edu_enc = education_mapping[education_level]

    # Create job title dummies
    job_title_data = {col: 0 for col in job_titles}
    if job_title != 'Others':
        job_title_data[job_title] = 1

    # Create the input dataframe for prediction
    input_df = pd.DataFrame({
        'Age': [age],
        'Gender': [gender_enc],
        'Education Level': [edu_enc],
        'Years of Experience': [years_of_exp],
        **job_title_data
    })

    # Predict salary
    pred_salary = model.predict(input_df)[0]

    st.success(f"Predicted Salary: ${pred_salary:,.2f}")
