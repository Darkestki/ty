import streamlit as st
import pickle
import numpy as np

# Load the saved model
try:
    with open('knn_best.sav', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    st.error("Model file 'knn_best.sav' not found. Please ensure the model file is in the same directory as the app.")
    model = None

st.title('Sales Prediction App (Optimized KNN)')

if model:
    st.write('Enter the advertising budgets for TV, Radio, and Newspaper to predict sales using the optimized KNN model.')

    # Create input fields
    tv_budget = st.number_input('TV Advertising Budget ($)', min_value=0.0, value=100.0)
    radio_budget = st.number_input('Radio Advertising Budget ($)', min_value=0.0, value=20.0)
    newspaper_budget = st.number_input('Newspaper Advertising Budget ($)', min_value=0.0, value=10.0)

    # Prediction button
    if st.button('Predict Sales'):
        # Create a numpy array from the inputs
        input_data = np.array([[tv_budget, radio_budget, newspaper_budget]])

        # Make prediction
        prediction = model.predict(input_data)

        # Display the prediction
        st.subheader('Predicted Sales')
        st.write(f'{prediction[0]:.2f} units')
else:
    st.warning("Cannot make predictions because the model file was not loaded.")
