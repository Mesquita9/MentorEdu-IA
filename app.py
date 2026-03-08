import streamlit as st
import pdfplumber, os, faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# --- 1. CONFIGURAÇÃO E ESTILO (DESIGN STARTUP/IFCE) ---
st.set_page_config(page_title="MentorEdu - Inércia Zero", page_icon="🧪", layout="wide")

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
    <style>
    /* Estilo Global */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    .stApp { background-color: #0b0e14; color: #f0f2f6; }
    
    /* Barra Lateral Profissional */
    [data-testid="stSidebar"] { 
        background-color: #161b22 !important; 
        border-right: 1px solid #30363d;
    }
    [data-testid="stSidebar"] .stMarkdown p { 
        color: #ffffff !important; 
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1rem !important; 
    }
    [data-testid="stSidebar"] h3 { color: #00d4ff !important; letter-spacing: -1px; }

    /* Título MentorEdu com Gradiente */
    .main-title { 
        text-align: center; 
        background: linear-gradient(90deg, #00d4ff, #88e23b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 3.5rem; 
        margin-top: -60px; letter-spacing: -2px;
    }
    .subtitle { 
        text-align: center; color: #9eaab7; 
        font-weight: 400; margin-top: -15px; margin-bottom: 2rem; 
    }

    /* Balões de Chat Glassmorphism */
    [data-testid="stChatMessage"] { 
        background-color: rgba(30, 37, 48, 0.7) !important; 
        border: 1px solid rgba(255, 255, 255, 0.1) !important; 
        border-radius: 12px !important;
        backdrop-filter: blur(10px);
        margin-bottom: 10px;
    }
    [data-testid="stChatMessage"] p { color: #ffffff !important; font-size: 1.05rem !important; }
    
    /* Nomes dos Personagens */
    .user-name, .bot-name {
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase; font-size: 0.75rem;
        letter-spacing: 1px; margin-bottom: 5px; display: block;
    }
    .user-name { color: #88e23b; }
    .bot-name { color: #00d4ff; }

    /* Barra de Input */
    .stChatInputContainer { background-color: #0b0e14 !important; }
    .stChatInputContainer div { border: 1px solid #30363d !important; border-radius: 10px !important; }
    
    /* Esconder elementos nativos */
    header, footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_models():
    # Garante que o app não quebre se a chave estiver faltando
    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key) if api_key else None
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return client, model

client, model = load_models()

# --- 2. BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=100)
    st.markdown("### 🧪 Projeto Inércia Zero")
    
    variante = st.selectbox(
        "Modo de Operação:",
        ["Rick Sarcástico", "Rick Acadêmico", "Rick Inércia Zero"]
    )
    
    st.markdown("---")
    uploaded_file = st.file_uploader("📂 Alimentar Cérebro (PDF)", type="pdf")
    
    if st.button("Resetar Realidade"):
        st.session_state.mensagens = []
        st.rerun()

# --- 3. PROCESSAMENTO RAG (INTELIGÊNCIA) ---
chunks, pages = [], []
if uploaded_file:
    with st.spinner("Rick extraindo dados do PDF..."):
        with pdfplumber.open(uploaded_file) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    for line in text.split('\n'):
                        line = line.strip()
                        if len(line) > 50:
                            chunks.append(line)
                            pages.append(i + 1)
        
        if chunks:
            embeddings = model.encode(chunks)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.
