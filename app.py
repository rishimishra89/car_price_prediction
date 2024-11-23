import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the saved model
with open('car_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define preprocessing pipeline
categorical_columns = ['fuel', 'seller_type', 'transmission', 'owner']
numerical_columns = ['year', 'km_driven']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(drop='first'), categorical_columns)
    ]
)

# Streamlit app title
st.title("Car Price Prediction App")

# Create input fields for user input
year = st.number_input("Year of the car (e.g., 2015):", min_value=1990, max_value=2023, step=1)
km_driven = st.number_input("Kilometers driven (e.g., 50000):", min_value=0)
fuel = st.selectbox("Fuel type:", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
seller_type = st.selectbox("Seller type:", ['Dealer', 'Individual'])
transmission = st.selectbox("Transmission type:", ['Manual', 'Automatic'])
owner = st.selectbox("Owner type:", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])

# Predict button
if st.button("Predict Selling Price"):
    # Create a DataFrame from user input
    input_data = pd.DataFrame(
        [[year, km_driven, fuel, seller_type, transmission, owner]],
        columns=['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']
    )
    
    # Preprocess the input data
    preprocessed_data = preprocessor.fit_transform(input_data)
    
    # Make prediction
    prediction = model.predict(preprocessed_data)
    
    # Display the predicted price
    st.success(f"Predicted Selling Price: ₹{prediction[0]:,.2f}")
