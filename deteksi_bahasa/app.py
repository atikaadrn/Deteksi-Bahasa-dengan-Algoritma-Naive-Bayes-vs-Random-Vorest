import streamlit as st
import pickle

# Load model & vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkll", "rb"))

# Judul aplikasi
st.title("Demo Analisis Sentimen dengan Deteksi Bahasa")

# Input user
user_input = st.text_input("Masukkan kalimat:")

# Prediksi
if st.button("Prediksi"):
    # Transform teks ke fitur numerik
    X = vectorizer.transform([user_input])

    # Prediksi pakai model
    prediction = model.predict(X)[0]

    st.success(f"Hasil prediksi: {prediction}")
