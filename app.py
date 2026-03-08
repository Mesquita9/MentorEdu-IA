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

    /* BARRA LATERAL (SIDEBAR) - RESOLVENDO AS CAIXAS BRANCAS */
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

    /* FIX PARA CAMPOS DE SELEÇÃO E UPLOADER (FUNDO ESCURO + LETRA BRANCA) */
    div[data-baseweb="select"] > div {
        background-color: #1e2530 !important;
        border: 2px solid #3b424b !important;
        color: white !important;
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

    /* BALÕES DE CONVERSA */
    [data-testid="stChatMessage"] {
        background-color: #1c2128 !important;
        border: 1px solid #30363d !important;
        border-radius: 12px;
        margin-bottom: 15px;
    }
    
    header, footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

# 2. MOTOR DE INTELIGÊNCIA
@st.cache_resource
def load_engine():
    try:
        api_key = os.getenv("GROQ_API_KEY")
        c = Groq(api_key=api_key)
        m = SentenceTransformer("all-MiniLM-L6-v2")
        return c, m
    except Exception:
        return None, None

client, model = load_engine()

# 3. BARRA LATERAL
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=120)
    else:
        st.markdown("## 🧪 MENTOREDU")
        
    st.markdown("---")
    modo = st.selectbox("ESTILO DO RICK:", 
                        ["Rick Acadêmico", "Rick Inércia Zero", "Rick Sarcástico"])
    
    up = st.file_uploader("📂 SUBIR BASE (PDF)", type="pdf")
    
    if st.button("LIMPAR HISTÓRICO"):
        st.session_state.mensagens = []
        st.rerun()

# 4. LÓGICA DE MEMÓRIA E PDF
if "mensagens" not in st.session_state:
