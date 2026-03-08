import streamlit as st
import pdfplumber, os, faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# --- 1. DESIGN DE ELITE (INTER & JETBRAINS MONO) ---
st.set_page_config(page_title="MentorEdu", page_icon="🧪", layout="wide")

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;800&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    .stApp { background-color: #0b0e14; color: #f0f2f6; }
    
    /* SIDEBAR LEGÍVEL */
    [data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    [data-testid="stSidebar"] .stMarkdown p { font-family: 'JetBrains Mono', monospace !important; font-size: 0.95rem !important; }

    /* TÍTULO GRADIENTE */
    .main-title { 
        text-align: center; background: linear-gradient(90deg, #00d4ff, #88e23b);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 3.5rem; margin-top: -60px; letter-spacing: -2px;
    }

    /* CHAT GLASSMORPHISM */
    [data-testid="stChatMessage"] { 
        background-color: rgba(30, 37, 48, 0.7) !important; 
        border: 1px solid rgba(255, 255, 255, 0.1) !important; 
        border-radius: 12px !important; backdrop-filter: blur(10px);
    }
    .name-tag { font-family: 'JetBrains Mono', monospace !important; font-size: 0.75rem; letter-spacing: 1px; display: block; margin-bottom: 5px; }
    header, footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_all():
    c = Groq(api_key=os.getenv("GROQ_API_KEY"))
    m = SentenceTransformer("all-MiniLM-L6-v2")
    return c, m

client, model = load_all()

# --- 2. BARRA LATERAL ---
with st.sidebar:
    if os.path.exists("logo.png"): st.image("logo.png", width=100)
    st.markdown("### 🧪 PROJETO INÉRCIA ZERO")
    var = st.selectbox("MODO:", ["Rick Sarcástico", "Rick Acadêmico", "Rick Inércia Zero"])
    up = st.file_uploader("📂 PDF PARA ANÁLISE", type="pdf")
    if st.button("RESETAR UNIVERSO"):
        st.session_state.mensagens = []
        st.rerun()

# --- 3. INTELIGÊNCIA RAG ---
chunks, pgs = [], []
if up:
    with st.spinner("Rick lendo..."):
        with pdfplumber.open(up) as pdf:
            for i, p in enumerate(pdf.pages):
                t = p.extract_text()
                if t:
                    for l in t.split('\n'):
                        if len(l.strip()) > 50:
                            chunks.append(l.strip()); pgs.append(i+1)
        if chunks:
            embs = model.encode(chunks)
            index = faiss.IndexFlatL2(embs.shape[1])
            index.add(np.array(embs))

# --- 4. INTERFACE ---
st.markdown('<h1 class="main-title">MentorEdu</h1>', unsafe_allow_html=True)
if "mensagens" not in st.session_state: st.session_state.mensagens = []

for m in st.session_state.mensagens:
    av = "logo2.png" if m["role"] == "user" else "logo.png"
