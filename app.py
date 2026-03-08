import streamlit as st
import pdfplumber, os, faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. Configuração e Estilo (Foco em Legibilidade Total)
st.set_page_config(page_title="Inércia Zero", page_icon="🧪", layout="wide")

st.markdown("""
    <style>
    /* Fundo Dark e Barra Lateral */
    .stApp { background-color: #0b0e14; color: #f0f2f6; }
    [data-testid="stSidebar"] { background-color: #1a1f26 !important; }
    
    /* Texto da Sidebar - FORÇAR BRANCO PARA FICAR LEGÍVEL */
    [data-testid="stSidebar"] * { color: #ffffff !important; font-weight: 500; }
    
    /* Título MentorEdu */
    .main-title { text-align: center; color: #00d4ff; font-weight: 800; font-size: 3rem; margin-top: -60px; }
    
    /* Balões de Chat - Estilo Premium */
    [data-testid="stChatMessage"] { 
        background-color: #1e2530 !important; 
        border: 1px solid #30363d !important; 
        border-radius: 15px !important;
    }
    [data-testid="stChatMessage"] p { color: #ffffff !important; font-size: 1.1rem !important; }

    /* Barra de Texto Fixa e Visível */
    .stChatInputContainer { background-color: #0b0e14 !important; }
    .stChatInputContainer div { background-color: #21262d !important; border: 1px solid #00d4ff !important; }
    
    /* Esconder Lixo */
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
    st.markdown("### 🧪 Projeto Inércia Zero") # Este texto agora está branco!
    var = st.selectbox("Variante:", ["Rick Sarcástico", "Rick Acadêmico", "Rick Inércia Zero"])
    up = st.file_uploader("📂 Subir PDF", type="pdf")
    if st.button("Limpar Histórico"):
        st.session_state.mensagens = []
        st.rerun()

# 3. Cérebro RAG
chunks, pgs = [], []
if up:
    with st.spinner("Lendo documento..."):
        with pdfplumber.open(up) as pdf:
            for i, p in enumerate(pdf.pages):
                txt = p.extract_text()
                if txt:
                    for l in txt.split('\n'):
                        if len(l.strip()) > 50:
                            chunks.append(l.strip()); pgs.append(i+1)
        if chunks:
            embs = model.encode(chunks)
            index = faiss.IndexFlatL2(embs.shape[1])
            index.add(np.array(embs))

# 4. Interface de Chat
st.markdown('<h1 class="main-title">MentorEdu</h1>', unsafe_allow_html=True)
if "mensagens" not in st.session_state: st.session_state.mensagens = []

for m in st.session_state.mensagens:
    av = "logo2.png" if m["role"] == "user" else "logo.png"
    with st.chat_message(m["role"], avatar=av):
        st.markdown(f"**{'Morty' if m['role'] == 'user' else 'Rick'}:**\n{m['content']}")

if prompt := st.chat_input("Diz aí, Morty..."):
    st.session_state.mensagens.append({"role": "user", "content": prompt
