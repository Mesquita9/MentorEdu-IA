import streamlit as st
import pdfplumber, os, faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. CONFIGURAÇÃO E DESIGN DE ALTO CONTRASTE
st.set_page_config(page_title="MentorEdu", page_icon="🧪", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    
    /* FUNDO E TEXTO GLOBAL */
    .stApp { background-color: #0e1117; color: #ffffff; font-family: 'Inter', sans-serif; }

    /* BARRA LATERAL (SIDEBAR) */
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    
    /* FORÇAR VISIBILIDADE DOS TEXTOS DA ESQUERDA */
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }

    /* CORREÇÃO DAS CAIXAS BRANCAS (SELECTBOX E UPLOADER) */
    div[data-baseweb="select"] > div {
        background-color: #1e2530 !important;
        border: 2px solid #3b424b !important;
    }
    
    div[data-testid="stFileUploader"] section {
        background-color: #1e2530 !important;
        border: 1px dashed #444c56 !important;
        color: white !important;
    }

    /* TÍTULO COM GRADIENTE */
    .main-title { 
        text-align: center; 
        background: linear-gradient(90deg, #00d4ff, #88e23b);
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 3.5rem; margin-top: -40px; 
    }

    /* BALÕES DE CONVERSA (DARK MODE) */
    [data-testid="stChatMessage"] {
        background-color: #1c2128 !important;
        border: 1px solid #30363d !important;
        border-radius: 12px;
        margin-bottom: 15px;
    }
    
    /* INPUT DE MENSAGEM */
    [data-testid="stChatInput"] {
        background-color: #0e1117 !important;
    }

    header, footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

# 2. CARREGAMENTO DE MODELOS (CACHEADO)
@st.cache_resource
def load_engine():
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("GROQ_API_KEY não encontrada nas variáveis de ambiente!")
        c = Groq(api_key=api_key)
        m = SentenceTransformer("all-MiniLM-L6-v2")
        return c, m
    except Exception as e:
