import streamlit as st
import pdfplumber
import os
import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# --- 1. CONFIGURAÇÃO BÁSICA ---
st.set_page_config(page_title="MentorEdu", page_icon="🧪", layout="wide")

# CSS super simplificado para forçar texto branco e fundo escuro
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    [data-testid="stSidebar"] { background-color: #161b22; }
    div[data-testid="stChatMessage"] { background-color: #1c2128; border: 1px solid #30363d; }
    p, span, div { color: white !important; }
</style>
""", unsafe_allow_html=True)

# --- 2. CARREGAR IA ---
@st.cache_resource
def load_engine():
    try:
        # Pega a chave do ambiente (Streamlit Secrets)
        c = Groq(api_key=os.getenv("GROQ_API_KEY"))
        m = SentenceTransformer("all-MiniLM-L6-v2")
        return c, m
    except Exception:
        return None, None

client, model = load_engine()

# --- 3. BARRA LATERAL ---
with st.sidebar:
    st.markdown("## 🧪 MENTOREDU")
    modo = st.selectbox("ESTILO DO RICK:", ["Rick Acadêmico", "Rick Sarcástico"])
    up = st.file_uploader("📂 SUBIR BASE (PDF)", type="pdf")
    
    if st.button("LIMPAR HISTÓRICO"):
        st.session_state.mensagens = []
        st.rerun()

# --- 4. MEMÓRIA E PDF ---
if "mensagens" not in st.session_state:
    st.
