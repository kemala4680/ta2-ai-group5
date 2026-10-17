import streamlit as st
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import altair as alt
import os

# Atur konfigurasi halaman web
st.set_page_config(page_title="Validator Pas Foto AI", page_icon="📸", layout="centered")

st.title("Validator Kelayakan Pas Foto AI 📸")
st.write("Unggah foto Anda untuk memverifikasi kesesuaian parameter standar pas foto menggunakan model Deep Learning.")

# Definisikan path model dan konfigurasi
MODEL_DIR = "/content/run_03_batchnorm/"
MODEL_PATH = os.path.join(MODEL_DIR, "model_pasfoto.keras")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

# Memuat model dan konfigurasi menggunakan cache agar tidak dimuat berulang kali
@st.cache_resource
def load_validator_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        st.error(f"Model tidak ditemukan di {MODEL_PATH}")
        return None

@st.cache_data
def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    else:
        # Default fallback attributes jika config.json tidak ada atau tidak sesuai
        return {
            "attributes": [
                "Background_Polos", "Kacamata_Gelap", "Penutup_Kepala", 
                "Pandangan_Lurus", "Mulut_Terbuka", "Wajah_Terlihat_Jelas", 
                "Pencahayaan_Baik", "Ketajaman_Cukup", "Posisi_Simetris", "Format_Sesuai"
            ]
        }

model = load_validator_model()
config = load_config()
attributes = config.get("attributes", [
    "Background_Polos", "Kacamata_Gelap", "Penutup_Kepala", 
    "Pandangan_Lurus", "Mulut_Terbuka", "Wajah_Terlihat_Jelas", 
    "Pencahayaan_Baik", "Ketajaman_Cukup", "Posisi_Simetris", "Format_Sesuai"
])

# Widget Pengunggah Gambar
uploaded_file = st.file_uploader("Pilih file gambar pas foto Anda...", type=["jpg", "jpeg", "png"])

def preprocess_image(pil_img):
    # Melakukan Center Cropping menjadi persegi
    width, height = pil_img.size
    min_dim = min(width, height)
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2
    cropped_img = pil_img.crop((left, top, right, bottom))
    
    # Mengubah ukuran ke 128x128 piksel
    resized_img = cropped_img.resize((128, 128), Image.Resampling.LANCZOS)
    
    # Normalisasi array gambar
    img_array = np.array(resized_img).astype('float32') / 255.0
    
    # Jika gambar greyscale, ubah menjadi RGB
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        # Hapus alpha channel jika ada
        img_array = img_array[:, :, :3]
        
    # Menambahkan dimensi batch
    img_array = np.expand_dims(img_array, axis=0)
    return cropped_img, resized_img, img_array

if uploaded_file is not None and model is not None:
    # Membuka gambar asli
    raw_image = Image.open(uploaded_file)
    
    # Preprocessing
    cropped_img, resized_img, processed_array = preprocess_image(raw_image)
    
    # Menampilkan Gambar di UI
    st.subheader("Pemrosesan Gambar (128x128)")
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image(raw_image, caption="Gambar Asli", use_container_width=True)
    with col_img2:
        st.image(resized_img, caption="Hasil Crop & Resize (128x128)", use_container_width=True)
    
    # Jalankan inferensi
    with st.spinner("Menganalisis pas foto dengan model AI..."):
        predictions = model.predict(processed_array)[0]
        
    # Menyusun data untuk visualisasi
    chart_data = pd.DataFrame({
        'Atribut': attributes[:len(predictions)],
        'Probabilitas (%)': [float(p * 100) for p in predictions]
    })
    
    # Membuat Bar Chart Interaktif dengan Altair
    st.subheader("Analisis Atribut Detail")
    chart = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X('Probabilitas (%)', scale=alt.Scale(domain=[0, 100])),
        y=alt.Y('Atribut', sort='-x', title='Kriteria Atribut'),
        color=alt.Color('Probabilitas (%)', scale=alt.Scale(scheme='viridis')),
        tooltip=['Atribut', 'Probabilitas (%)']
    ).properties(
        width=600,
        height=350
    )
    
    st.altair_chart(chart, use_container_width=True)
    
    # Tampilkan tabel data
    with st.expander("Lihat Data Angka Atribut"):
        st.dataframe(chart_data.style.format({'Probabilitas (%)': '{:.2f}%'}))

elif model is None:
    st.warning("Aplikasi belum siap karena model gagal dimuat. Pastikan file model ada di direktori yang sesuai.")
else:
    st.info("Silakan unggah pas foto untuk memulai proses analisis AI.")
