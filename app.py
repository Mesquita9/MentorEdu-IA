import streamlit as st
import pdfplumber
import os
import numpy as np
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. Configuração da Página
st.set_page_config(page_title="Inércia Zero - MentorEdu", page_icon="🧪", layout="wide")

# 2. CSS Estilo Dark Mode Refinado (Gemini/GPT Style)
st.markdown("""
    <style>
    /* Fundo principal e Sidebar */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
    }
    
    /* Título MentorEdu Neon */
    .main-title {
        text-align: center;
        color: #88e23b; 
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        margin-top: -60px;
    }
    
    /* Subtítulo */
    .subtitle {
        text-align: center;
        color: #9eaab7;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Estilização dos Balões de Chat */
    [data-testid="stChatMessage"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 12px;
        margin-bottom: 10px;
        padding: 15px;
    }

    /* Fixar e Estilizar a Barra de Texto (Input) */
    .stChatInputContainer {
        background-color: #0e1117 !important;
        padding-bottom: 20px;
    }
    .stChatInputContainer > div {
        background-color: #21262d !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
    }
    textarea {
        color: #ffffff !important;
    }

    /* Esconder elementos desnecessários do Streamlit */
    header { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

# --- Inicialização de Recursos ---
@st.cache_resource
def load_resources():
    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key) if api_key else None
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return client, model

client, model = load_resources()

# --- BARRA LATERAL (Painel de Controle) ---
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=150) # Rick
    
    st.markdown("### 🧪 Projeto Inércia Zero")
    
    variante = st.selectbox(
        "Escolha a variante do Rick:",
        ["Rick Sarcástico", "Rick Acadêmico", "Rick Inércia Zero"]
    )
    
    st.info("Morty, coloca o PDF aqui ou ficaremos presos nessa dimensão!")
    uploaded_file = st.file_uploader("📂 Subir PDF", type="pdf", label_visibility="collapsed")
    
    if st.button("Explodir Histórico (Reset)"):
        st.session_state.mensagens = []
        st.rerun()

# --- Processamento RAG (Cérebro do Rick) ---
chunks, paginas = [], []
if uploaded_file:
    with st.spinner("Rick está analisando... isso é ciência, Morty!"):
        with pdfplumber.open(uploaded_
