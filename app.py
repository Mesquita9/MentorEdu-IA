import streamlit as st
import pdfplumber, os, faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# --- CONFIGURAÇÃO E ESTILO DARK ---
st.set_page_config(page_title="Inércia Zero", page_icon="🧪", layout="wide")

st.markdown("""
    <style>
    /* Fundo Escuro Total */
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    [data-testid="stSidebar"] { background-color: #161b22 !important; }
    
    /* Título MentorEdu */
    .main-title { text-align: center; color: #88e23b; font-weight: 800; font-size: 3rem; margin-top: -60px; }
    
    /* Balões de Chat (Dark) */
    [data-testid="stChatMessage"] { background-color: #161b22 !important; border: 1px solid #30363d !important; border-radius: 12px; }
    
    /* Barra de Texto (Fixa e Escura) */
    .stChatInputContainer { background-color: #0e1117 !important; padding-bottom: 20px; }
    .stChatInputContainer div { background-color: #21262d !important; border: 1px solid #30363d !important; color: white !important; }
    
    /* Esconder Lixo do Streamlit */
    header, footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def init():
    c = Groq(api_key=os.getenv("GROQ_API_KEY"))
    m = SentenceTransformer("all-MiniLM-L6-v2")
    return c, m

client, model = init()

# --- SIDEBAR ---
with st.sidebar:
    if os.path.exists("logo.png"): st.image("logo.png", width=120)
    st.title("🧪 Painel")
    var = st.selectbox("Variante:", ["Rick Sarcástico", "Rick Acadêmico", "Rick Inércia Zero"])
    up = st.file_uploader("Subir PDF", type="pdf")
    if st.button("Resetar"):
        st.session_state.mensagens = []
        st.rerun()

# --- RAG (CÉREBRO) ---
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

# --- INTERFACE ---
st.markdown('<h1 class="main-title">MentorEdu</h1>', unsafe_allow_html=True)
if "mensagens" not in st.session_state: st.session_state.mensagens = []

for m in st.session_state.mensagens:
    av = "logo2.png" if m["role"] == "user" else "logo.png"
    with st.chat_message(m["role"], avatar=av): st.markdown(m["content"])

if prompt := st.chat_input("Diz aí, Morty..."):
    st.session_state.mensagens.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="logo2.png"): st.markdown(prompt)

    with st.chat_message("assistant", avatar="logo.png"):
        ctx = ""
        if up and chunks:
            q_emb = model.encode([prompt])
            D, I = index.search(np.array(q_emb), k=2)
            for idx in I[0]: ctx += f"[Pág {pgs[idx]}] {chunks[idx]}\
