import streamlit as st
import pdfplumber
import os
import numpy as np
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. Configuração da Página
st.set_page_config(page_title="Inércia Zero - MentorEdu", page_icon="🧪", layout="wide")

# 2. CSS Estilo Dark Mode (Inspirado em Gemini/GPT)
st.markdown("""
    <style>
    /* Fundo Escuro Total */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    /* Estilo da Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
    }
    /* Título e Subtítulo */
    .main-title {
        text-align: center;
        color: #88e23b; 
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        margin-top: -50px;
    }
    .subtitle {
        text-align: center;
        color: #9eaab7;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    /* Balões de Chat */
    [data-testid="stChatMessage"] {
        background-color: #21262d;
        border: 1px solid #30363d;
        border-radius: 12px;
        margin-bottom: 10px;
    }
    /* Forçar a cor do texto no Input */
    input {
        color: white !important;
    }
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

# --- BARRA LATERAL ---
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=150)
    
    st.markdown("### 🧪 Projeto Inércia Zero")
    variante = st.selectbox(
        "Variante do Rick:",
        ["Rick Sarcástico", "Rick Acadêmico", "Rick Inércia Zero"]
    )
    
    uploaded_file = st.file_uploader("📂 PDF (opcional)", type="pdf")
    
    if st.button("Resetar Dimensão"):
        st.session_state.mensagens = []
        st.rerun()

# --- Processamento RAG ---
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

# --- Área de Chat ---
st.markdown('<h1 class="main-title">MentorEdu</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Projeto Inércia Zero - Escolha sua realidade</p>', unsafe_allow_html=True)

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

# Mostrar Histórico
for msg in st.session_state.mensagens:
    avatar_img = "logo2.png" if msg["role
