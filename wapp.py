import streamlit as st
import joblib 
import pandas as pd
import numpy as np


scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

st.title('Car Purchasing Power Estimator')

st.divider()
st.write('Please enter the following details to estimate the car purchasing power:')



age =st.number_input('Age',min_value=18,max_value=90,value=40, step=1)
annual_salary = st.number_input('Annual Salary', min_value=500, max_value=9999999999, value=12000, step=5000)
net_worth = st.number_input('Net Worth', min_value=0, max_value=999999999, step=2000, value=100000)


X = [age,annual_salary,net_worth]
calculate =st.button('Calculate')
st.divider()

X_scaled = scaler.transform([X])

if calculate:
    st.balloons()
    X_2 = np.array(X)
    X_array = scaler.transform([X_2])
    
    prediction = model.predict(X_array)
    
    # Check if prediction is less than 0
    if prediction[0] < 0:
        st.write("Prediction is: Negative")
    else:
        st.write(f"Prediction is: {prediction[0]:}")
    
    st.write("Advice: cars in the similar values")
       
else:
    st.write("Enter values")

      
    


