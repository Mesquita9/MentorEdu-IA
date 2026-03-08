import streamlit as st
import pdfplumber
import os
import numpy as np
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. Configuração da Página
st.set_page_config(page_title="Inércia Zero - MentorEdu", page_icon="🧪", layout="wide")

# --- ESTILIZAÇÃO DINÂMICA (Dark/Light) ---
if "tema_escuro" not in st.session_state:
    st.session_state.tema_escuro = True

# Cores baseadas no modo
bg_color = "#0e1117" if st.session_state.tema_escuro else "#ffffff"
text_color = "#ffffff" if st.session_state.tema_escuro else "#000000"
card_bg = "#1f2937" if st.session_state.tema_escuro else "#f0f2f6"

st.markdown(f"""
    <style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    .main-title {{
        text-align: center;
        color: #97ce4c; 
        font-family: 'Courier New', monospace;
        font-weight: 900;
        font-size: 3.5rem;
    }}
    .stChatMessage {{
        background-color: {card_bg};
        border: 2px solid #97ce4c;
    }}
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

# --- BARRA LATERAL (Portal de Comando) ---
with st.sidebar:
    st.image("logo.png", width=150)
    st.markdown("### ⚙️ Ajustes de Dimensão")
    
    # Alternador de Tema
    st.session_state.tema_escuro = st.toggle("Modo Escuro", value=True)
    
    # Seletor de Personalidade
    persona = st.selectbox(
        "Variante do Rick:",
        ["Rick Sarcástico", "Rick Acadêmico (IFCE)", "Rick Motivador"]
    )
    
    st.markdown("---")
    st.info("Morty, joga o PDF aí!")
    uploaded_file = st.file_uploader("Upload", type="pdf", label_visibility="collapsed")
    
    if st.button("Limpar Linha do Tempo"):
        st.session_state.mensagens = []
        st.rerun()

# --- Lógica de Personalidade ---
dicionario_personas = {
    "Rick Sarcástico": "Você é o Rick Sanchez clássico. Genial, sarcástico e chama o usuário de Morty. Use 'Wubba Lubba Dub Dub'.",
    "Rick Acadêmico (IFCE)": "Você é uma variante do Rick que é Diretor de Pesquisa no IFCE. Você é brilhante, usa termos científicos de alto nível e foca em normas técnicas, mas ainda é superior.",
    "Rick Motivador": "Você é o Rick focado no Projeto Inércia Zero. Seu objetivo é tirar o Morty da preguiça. Seja agressivo e motivador para ele terminar a pesquisa."
}

# --- PROCESSAMENTO PDF (RAG) ---
chunks, paginas = [], []
if uploaded_file:
    with st.spinner("Lendo...
