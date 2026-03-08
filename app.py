import streamlit as st
import pdfplumber
import os
import numpy as np
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. Configuração da Página
st.set_page_config(page_title="Inércia Zero - MentorEdu", page_icon="🧪", layout="wide")

# 2. CSS Estilo Dark (Inspirado em Gemini/GPT)
st.markdown("""
    <style>
    /* Forçar Fundo Escuro em Tudo */
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    section[data-testid="stSidebar"] { background-color: #161b22 !important; }
    
    /* Título MentorEdu Neon */
    .main-title { text-align: center; color: #88e23b; font-family: 'Inter', sans-serif; font-weight: 800; font-size: 3rem; margin-top: -50px; }
    .subtitle { text-align: center; color: #9eaab7; font-size: 1.1rem; margin-bottom: 2rem; }

    /* Balões de Chat Estilo Grafite */
    [data-testid="stChatMessage"] { background-color: #21262d; border: 1px solid #30363d; border-radius: 12px; margin-bottom: 10px; color: white !important; }
    
    /* Garantir que o Input de Texto apareça no rodapé */
    .stChatInputContainer { padding-bottom: 20px; background-color: transparent !important; }
    input { color: #ffffff !important; }
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

# --- BARRA LATERAL (Portal de Comando) ---
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=150) # Rick
    
    st.markdown("### 🧪 Projeto Inércia Zero")
    variante = st.selectbox("Variante do Rick:", ["Rick Sarcástico", "Rick Acadêmico", "Rick Inércia Zero"])
    
    uploaded_file = st.file_uploader("📂 PDF para análise", type="pdf")
    
    if st.button("Explodir Dimensão (Reset)"):
        st.session_state.mensagens = []
        st.rerun()

# --- Cérebro do Rick (RAG) ---
chunks, paginas = [], []
if uploaded_file:
    with st.spinner("Rick está lendo..."):
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

# --- Interface Principal ---
st.markdown('<h1 class="main-title">MentorEdu</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Projeto Inércia Zero - Rompendo a estática acadêmica</p>', unsafe_allow_html=True)

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

# Exibição do Histórico
for msg in st.session_state.mensagens:
    avatar = "logo2.png" if msg["role"] == "user" else "logo.png" # Morty e Rick
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# --- BARRA DE TEXTO (AQUI ESTÁ ELA, MORTY!) ---
if prompt := st.chat_input("Diz aí, Morty..."):
    st.session_state.mensagens.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="logo2.png"): # Morty
        st.markdown(prompt
