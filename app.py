import streamlit as st
import pdfplumber
import os
import numpy as np
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. Configuração da Página
st.set_page_config(page_title="Inércia Zero - MentorEdu", page_icon="🧪", layout="wide")

# 2. Estilização Rick and Morty
st.markdown("""
    <style>
    .main-title { text-align: center; color: #97ce4c; font-family: 'Courier New', monospace; font-weight: 900; font-size: 3.5rem; text-shadow: 2px 2px #44281d; }
    .subtitle { text-align: center; color: #88e23b; font-size: 1.2rem; margin-bottom: 2rem; }
    .stChatMessage { border-radius: 15px; border: 2px solid #97ce4c; }
    </style>
    """, unsafe_allow_html=True)

# --- Inicialização ---
@st.cache_resource
def load_resources():
    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key) if api_key else None
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return client, model

client, model = load_resources()

# --- BARRA LATERAL ---
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=150)
    
    st.markdown("### 🧪 Projeto Inércia Zero")
    
    # SELETOR DE PERSONALIDADE
    variante = st.selectbox(
        "Escolha a variante do Rick:",
        ["Rick Sarcástico", "Rick Acadêmico", "Rick Inércia Zero"]
    )
    
    st.info("Morty, coloca o PDF aqui ou ficaremos presos nessa aula para sempre!")
    uploaded_file = st.file_uploader("Subir PDF", type="pdf", label_visibility="collapsed")
    
    if st.button("Resetar Dimensão"):
        st.session_state.mensagens = []
        st.rerun()

# --- Cérebro do Rick (RAG) ---
chunks, paginas = [], []
if uploaded_file:
    with st.spinner("Analisando... isso é ciência de verdade, Morty!"):
        with pdfplumber.open(uploaded_file) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    for line in text.split('\n'):
                        if len(line.strip()) > 50:
                            chunks.append(line.strip())
                            paginas.append(i + 1)
        if chunks:
            embeddings = model.encode(chunks)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings))

# --- Interface de Chat ---
st.markdown('<h1 class="main-title">MentorEdu</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Projeto Inércia Zero - Escolha sua realidade</p>', unsafe_allow_html
