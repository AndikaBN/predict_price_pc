import streamlit as st
import pandas as pd
import pickle

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('laptop_data.csv')

# Load model
@st.cache_resource
def load_model():
    with open('laptop_model.pkl', 'rb') as file:
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
