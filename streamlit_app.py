import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
from PIL import ImageOps
import numpy as np

st.set_page_config(
    page_title="Sistem Klasifikasi Penyakit Paru",
    page_icon="https://cdn3.iconfinder.com/data/icons/pandemic-solid-keep-healthy-routine/512/Virus_lung_infection-256.png"  # Anda juga dapat menggunakan path ke file icon, misalnya "icon.png"
)

# Fungsi untuk memuat model CNN
@st.cache_resource
def load_cnn_model(model_path):
    model = load_model(model_path)
    return model

# Fungsi untuk memproses dan memprediksi gambar
def predict_image(model, image, target_size):
    # Resize gambar ke ukuran yang diterima model
    grayscale_image = ImageOps.grayscale(image)
    image = grayscale_image.convert('RGB')
    image = image.resize(target_size)
    image_array = img_to_array(image) / 255.0  # Normalisasi
    image_array = np.expand_dims(image_array, axis=0)  # Tambahkan dimensi batch
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions)
    return predicted_class

# Konfigurasi Streamlit
st.title("Aplikasi Deteksi Penyakit Pada Paru Paru menggunakan CNN")

# Upload gambar
uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    image = Image.open(uploaded_file)  # Buka gambar

    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # Load model
    model_path = "model4parucnn.keras"  # Ganti dengan path model Anda
    model = load_cnn_model(model_path)
 
    # Lakukan prediksi
    target_size = (48, 48)  # Sesuaikan dengan model Anda
    predictions = predict_image(model, image, target_size)

    label_names = {0:'Paru Paru Anda Terkena Covid',1:'Paru Paru Anda NORMAL', 2:'Paru Paru Anda Terkena Pneumonia', 3:'Paru Paru Anda Terkena Tuberculosis'}
    predicted_label = label_names[predictions]

    
    # Tampilkan hasil prediksi
    st.write("Hasil Prediksi:")
    st.write(predicted_label)  # Ganti sesuai interpretasi output model Anda
