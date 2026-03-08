import streamlit as st
import pdfplumber, os, faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. Configuração e Estilo Dark Profissional (Contraste Máximo)
st.set_page_config(page_title="Inércia Zero", page_icon="🧪", layout="wide")

st.markdown("""
    <style>
    /* Fundo estilo Gemini/GPT */
    .stApp { background-color: #0b0e14; color: #f0f2f6; }
    [data-testid="stSidebar"] { background-color: #1a1f26 !important; }
    
    /* Título Neon */
    .main-title { text-align: center; color: #00d4ff; font-weight: 800; font-size: 3rem; margin-top: -60px; }
    .subtitle { text-align: center; color: #9eaab7; margin-bottom: 2rem; }

    /* Balões de Chat - AGORA LEGÍVEIS */
    [data-testid="stChatMessage"] { 
        background-color: #1e2530 !important; 
        border: 1px solid #30363d !important; 
        border-radius: 15px !important;
        padding: 20px !important;
        margin-bottom: 15px !important;
    }

    /* Forçar cor do texto nas mensagens para branco puro */
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] div {
        color: #ffffff !important;
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
    }

    /* Cores dos nomes para identificação rápida */
    .user-name { color: #88e23b; font-weight: bold; }
    .bot-name { color: #00d4ff; font-weight: bold; }

    /* Barra de Texto (Input) */
    .stChatInputContainer { background-color: #0b0e14 !important; }
    .stChatInputContainer div { 
        background-color: #21262d !important; 
        border: 1px solid #444c56 !important; 
    }
    textarea { color: white !important; }

    /* Esconder o que não interessa */
    header, footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_all():
    api_key = os.getenv("GROQ_API_KEY")
    c = Groq(api_key=api_key) if api_key else None
    m = SentenceTransformer("all-MiniLM-L6-v2")
    return c, m

client, model = load_all()

# 2. Painel Lateral (Sidebar)
with st.sidebar:
    if os.path.exists("logo.png"): 
        st.image("logo.png", width=120)
    st.markdown("### 🧪 Projeto Inércia Zero")
    var = st.selectbox("Variante do Rick:", ["Rick Sarcástico", "Rick Acadêmico", "Rick Inércia Zero"])
    up = st.file_uploader("📂 Subir PDF", type="pdf")
    if st.button("Resetar Universo"):
        st.session_state.mensagens = []
        st.rerun()

# 3. Cérebro RAG (Processamento de PDF)
chunks, pgs = [], []
if up:
    with st.spinner("Rick está analisando o documento..."):
        with pdfplumber.open(up) as pdf:
            for i, p in enumerate(pdf.pages):
                txt = p.extract_text()
                if txt:
                    for l in txt.split('\n'):
                        if len(
