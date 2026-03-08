import streamlit as st
import pdfplumber, os, faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. Configuração e Estilo Dark Profissional
st.set_page_config(page_title="Inércia Zero", page_icon="🧪", layout="wide")

st.markdown("""
    <style>
    /* Cores principais estilo Gemini/GPT */
    .stApp { background-color: #0b0e14; color: #f0f2f6; }
    [data-testid="stSidebar"] { background-color: #1a1f26 !important; }
    
    /* Título MentorEdu Neon */
    .main-title { text-align: center; color: #00d4ff; font-weight: 800; font-size: 3rem; margin-top: -60px; text-shadow: 0 0 10px #00d4ff44; }
    .subtitle { text-align: center; color: #9eaab7; margin-bottom: 2rem; }

    /* Balões de Chat Ultra Legíveis */
    [data-testid="stChatMessage"] { 
        background-color: #1e2530 !important; 
        border: 1px solid #30363d !important; 
        border-radius: 15px !important;
        color: #ffffff !important;
        font-size: 1.05rem;
    }
    
    /* Texto em negrito dentro do chat */
    strong { color: #88e23b !important; }

    /* Barra de Texto (Input) - Estilo Dark */
    .stChatInputContainer { background-color: #0b0e14 !important; padding-bottom: 20px; }
    .stChatInputContainer div { background-color: #21262d !important; border: 1px solid #00d4ff !important; border-radius: 12px !important; }
    textarea { color: white !important; }

    /* Esconder o lixo do Streamlit */
    header, footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_all():
    c = Groq(api_key=os.getenv("GROQ_API_KEY"))
    m = SentenceTransformer("all-MiniLM-L6-v2")
    return c, m

client, model = load_all()

# 2. Painel Lateral
with st.sidebar:
    if os.path.exists("logo.png"): st.image("logo.png", width=120)
    st.markdown("### 🧪 Projeto Inércia Zero")
    var = st.selectbox("Variante do Rick:", ["Rick Sarcástico", "Rick Acadêmico", "Rick Inércia Zero"])
    up = st.file_uploader("📂 PDF para o Rick ler", type="pdf")
    if st.button("Limpar Realidade"):
        st.session_state.mensagens = []
        st.rerun()

# 3. Cérebro RAG
chunks, pgs = [], []
if up:
    with st.spinner("Rick lendo..."):
        with pdfplumber.open(up) as pdf:
            for i, p in enumerate(pdf.pages
