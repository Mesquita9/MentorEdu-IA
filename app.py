import streamlit as st
import pdfplumber, os, faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. Configuração e Estilo (Design de Startup Tecnológica)
st.set_page_config(page_title="MentorEdu", page_icon="🧪", layout="wide")

# CSS em bloco único para evitar vazamento na tela
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    .stApp { background-color: #0b0e14; color: #f0f2f6; }
    
    /* Sidebar com contraste real */
    [data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    
    /* Título com Gradiente Premium */
    .main-title { 
        text-align: center; 
        background: linear-gradient(90deg, #00d4ff, #88e23b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 3.5rem; margin-top: -60px;
    }
    
    /* Chat Moderno (Glassmorphism) */
    [data-testid="stChatMessage"] { 
        background-color: rgba(30, 37, 48, 0.8) !important; 
        border: 1px solid #30363d !important; 
        border-radius: 15px !important;
        margin-bottom: 15px !important;
    }
    [data-testid="stChatMessage"] p { color: #ffffff !important; font-size: 1.1rem !important; }

    /* Barra de entrada */
    .stChatInputContainer { background-color: #0b0e14 !important; }
    .stChatInputContainer div { border: 1px solid #00d4ff !important; border-radius: 10px !important; }
    
    header, footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def carregar_tudo():
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return client, model

client, model = carregar_tudo()

# 2. Barra Lateral
with st.sidebar:
    if os.path.exists("logo.png"): st.image("logo.png", width=120)
    st.markdown("### 🧪 Projeto Inércia Zero")
    modo = st.selectbox("Personalidade do Rick:", ["Rick Acadêmico", "Rick Inércia Zero", "Rick Sarcástico"])
    pdf_file = st.file_uploader("📂 Base de Conhecimento (PDF)", type="pdf")
    if st.button("Resetar Sistema"):
        st.session_state.mensagens = []
        st.rerun()

# 3. Inteligência RAG
chunks, pgs = [], []
if pdf_file:
    with st.spinner("Analisando dados..."):
        with pdfplumber.open(pdf_file) as pdf:
            for i, p in enumerate(pdf.pages):
                txt = p.extract_text()
                if txt:
                    for l in txt.split('\n'):
                        if len(l.strip()) > 50:
                            chunks.append(l.strip())
                            pgs.append(i+1)
        if chunks:
            embeddings = model.encode(chunks)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index
