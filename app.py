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

    /* Forçar cor do texto nas mensagens */
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] div {
        color: #ffffff !important;
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
    }

    /* Destaque para os nomes Rick e Morty */
    .user-name { color: #88e23b; font-weight: bold; }
    .bot-name { color: #00d4ff; font-weight: bold; }

    /* Barra de Texto (Input) */
    .stChatInputContainer { background-color: #0b0e14 !important; }
    .stChatInputContainer div { background-color: #21262d !important; border: 1px solid #00d4ff !important; }
    textarea { color: white !important; }

    /* Esconder o que não interessa */
    header, footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_all():
    c = Groq(api_key=os.getenv("GROQ_API_KEY"))
    m = SentenceTransformer("all-MiniLM-L6-v2")
    return c, m

client, model = load_all()

# 2. Sidebar
with st.sidebar:
    if os.path.exists("logo.png"): st.image("logo.png", width=120)
    st.markdown("### 🧪 Projeto Inércia Zero")
    var = st.selectbox("Variante do Rick:", ["Rick Sarcástico", "Rick Acadêmico", "Rick Inércia Zero"])
    up = st.file_uploader("📂 PDF para análise", type="pdf")
    if st.button("Resetar Universo"):
        st.session_state.mensagens = []
        st.rerun()

# 3. Cérebro RAG
chunks, pgs = [], []
if up:
    with st.spinner("Rick lendo..."):
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

# 4. Interface Principal
st.markdown('<h1 class="main-title">MentorEdu</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Rompendo a inércia acadêmica</p>', unsafe_allow_html=True)

if "mensagens" not in st.session_state: st.session_state.mensagens = []

for m in st.session_state.mensagens:
    av = "logo2.png" if m["role"] == "user" else "logo.png"
    name = "Morty" if m["role"] == "user" else "Rick"
    cls = "user-name" if m["role"] == "user" else "bot-name"
    with st.chat_message(m["role"], avatar=av):
        st.markdown(f"<span class='{cls}'>{name}:</span>", unsafe_allow_html=True)
        st.markdown(m["content"])

if prompt := st.chat_input("Diz aí, Morty..."):
    st.session_state.mensagens.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="logo2.png"):
        st.markdown("<span class='user-name'>Morty:</span>", unsafe_allow_html=True)
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="logo.png"):
        ctx = ""
        if up and chunks:
            q_emb = model.encode([prompt])
            D, I = index.search(np.array(q_emb), k=2)
            for idx in I[0]: ctx += f"[Pág {pg
