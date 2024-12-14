import streamlit as st
import pandas as pd
import pickle
import requests
import os

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('https://raw.githubusercontent.com/AndikaBN/predict_price_pc/refs/heads/main/laptop_data.csv')

# Download dan load model
@st.cache_resource
def load_model():
    url = 'https://drive.google.com/uc?export=download&id=FILE_ID'  # Replace FILE_ID
    model_path = 'laptop_model.pkl'

    # Unduh model jika belum ada
    if not os.path.exists(model_path):
        with open(model_path, 'wb') as f:
            response = requests.get(url)
            f.write(response.content)

    # Load model dari file lokal
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Fungsi utama aplikasi
def main():
    st.title('Laptop Price Prediction')

    # Load data
    data = load_data()

    # Sidebar untuk input fitur
    st.sidebar.header('Input Features')

    def user_input_features():
        features = {}
        for col in data.columns[:-1]:  # Exclude the target column
            if data[col].dtype == 'object':
                features[col] = st.sidebar.selectbox(f'Select {col}', data[col].unique())
            else:
                features[col] = st.sidebar.number_input(f'Enter {col}', min_value=float(data[col].min()), max_value=float(data[col].max()))
        return pd.DataFrame(features, index=[0])

    input_df = user_input_features()

    # Tampilkan input fitur
    st.write('### Input Features')
    st.write(input_df)

    # Load model
    model = load_model()

    # Prediksi harga
    if st.button('Predict'):
        prediction = model.predict(input_df)
        formatted_prediction = f'$ {prediction[0]:,.2f}'
        st.write('### Predicted Laptop Price')
        st.write(formatted_prediction)

if __name__ == '__main__':
    main()
