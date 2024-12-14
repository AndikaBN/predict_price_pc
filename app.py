import streamlit as st
import pandas as pd
import pickle
import requests
# import sklearn
import os

@st.cache_data
def load_data():
    return pd.read_csv('https://raw.githubusercontent.com/AndikaBN/predict_price_pc/refs/heads/main/laptop_data.csv')

@st.cache_resource
def load_model():
    url = 'https://drive.google.com/uc?id=1ktcqRauXebQCm32e77KYAdSMc4gJTiBb'
    model_path = 'laptop_model.pkl'

    if not os.path.exists(model_path):
        response = requests.get(url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
        st.write('Model downloaded successfully.')

    st.write(f'Loading model from {model_path}...')
    with open(model_path, 'rb') as file:
        try:
            model = pickle.load(file)
            st.write('Model loaded successfully.')
        except Exception as e:
            st.error(f'Error loading model: {e}')
            raise
    return model


def main():
    st.title('Laptop Price Prediction')

    data = load_data()

    st.sidebar.header('Input Features')

    def user_input_features():
        features = {}
        for col in data.columns[:-1]:
            if data[col].dtype == 'object': 
                features[col] = st.sidebar.selectbox(f'Select {col}', data[col].unique())
            else:  
                features[col] = st.sidebar.number_input(
                    f'Enter {col}',
                    min_value=float(data[col].min()),
                    max_value=float(data[col].max())
                )
        return pd.DataFrame(features, index=[0])

    input_df = user_input_features()

    st.write('### Input Features')
    st.write(input_df)

    model = load_model()

    if st.button('Predict'):
        prediction = model.predict(input_df)
        formatted_prediction = f'$ {prediction[0]:,.2f}'
        st.write('### Predicted Laptop Price')
        st.write(formatted_prediction)

if __name__ == '__main__':
    main()
