import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import streamlit as st
import config
import folium
from streamlit_folium import st_folium

# === Konfigurasi Gemini ===
genai.configure(api_key=config.GOOGLE_API_KEY)

# === Load data ===
@st.cache_data
def load_data():
    df = pd.read_pickle("wisata_meta.pkl")
    return df

df = load_data()

# === Fungsi untuk generate jawaban pakai Gemini ===
def generate_answer(context, question, model="gemini-2.0-flash-lite"):
    prompt = f"""
Konteks:
{context}

Pertanyaan:
{question}

Jawablah dengan bahasa Indonesia yang jelas dan informatif, berdasarkan konteks di atas.
"""
    model_gemini = genai.GenerativeModel(model)
    response = model_gemini.generate_content(prompt)
    return response.text


# === Streamlit UI ===
st.title("🧭 Asisten AI Rekomendasi Wisata Religi Yogyakarta")
st.write("Tanyakan apa saja tentang tempat wisata religi di Yogyakarta!")

# Gunakan session_state agar hasil tidak hilang
if "answer" not in st.session_state:
    st.session_state.answer = None
    st.session_state.top_places = None

question = st.text_input("Masukkan pertanyaanmu:")

if st.button("Cari Jawaban"):
    if not question.strip():
        st.warning("Masukkan pertanyaan terlebih dahulu!")
    else:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        q_emb = embedder.encode([question])

        sims = cosine_similarity(q_emb, np.vstack(df["embedding"]))
        top_idx = np.argsort(sims[0])[::-1][:3]
        context = "\n\n".join(df.iloc[i]["deskripsi_singkat"] for i in top_idx)

        with st.spinner("🧠 Menghasilkan jawaban dengan Gemini..."):
            answer = generate_answer(context, question)

        st.session_state.answer = answer
        st.session_state.top_places = df.iloc[top_idx]


# === Tampilkan hasil kalau sudah ada di session_state ===
if st.session_state.answer:
    st.subheader("💬 Jawaban:")
    st.write(st.session_state.answer)

    top_places = st.session_state.top_places
    st.subheader("📍 Tempat wisata terkait:")
    for _, row in top_places.iterrows():
        st.write(f"**{row['nama_destinasi']}** — Koordinat: ({row['latitude']}, {row['longitude']})")

    # === Tambahkan peta interaktif ===
    st.subheader("🗺️ Peta Lokasi Wisata")

    center_lat = top_places["latitude"].mean()
    center_lon = top_places["longitude"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

    for _, row in top_places.iterrows():
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=f"{row['nama_destinasi']}",
            tooltip=row["nama_destinasi"],
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(m)

    st_folium(m, width=700, height=500)
