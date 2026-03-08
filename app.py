import streamlit as st
import pdfplumber, os, faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. CONFIGURAÇÃO DE TEMA (FORÇANDO CONTRASTE)
st.set_page_config(page_title="MentorEdu", layout="wide")

st.markdown("""
    <style>
    /* Força o fundo escuro e texto branco em TUDO */
    .stApp, [data-testid="stSidebar"], .stMarkdown, p, label {
        background-color: #0e1117 !important;
        color: #ffffff !important;
    }
    /* Estiliza as caixas de input que estavam brancas */
    div[data-baseweb="select"] > div, div[data-testid="stFileUploader"] section {
        background-color: #1c2128 !important;
        border: 1px solid #ffffff !important;
        color: white !important;
    }
    /* Título com cor visível */
    .titulo { color: #00d4ff; text-align: center; font-size: 40px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 2. INICIALIZAÇÃO
@st.cache_resource
def load_models():
    try:
        # Tenta pegar a chave secreta do Streamlit
        key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        return Groq(api_key=key), SentenceTransformer("all-MiniLM-L6-v2")
    except: return None, None

client, model = load_models()

# 3. BARRA LATERAL
with st.sidebar:
    st.markdown("### 🧪 CONFIGURAÇÃO")
    # Personalidades atualizadas conforme sua regra
    modo = st.selectbox("QUEM É O RICK?", ["Rick Acadêmico", "Rick Sarcástico"])
    up = st.file_uploader("SUBIR PDF", type="pdf")
    if st.button("LIMPAR CHAT"):
        st.session_state.chat = []
        st.rerun()

# 4. LÓGICA DO PDF
if "chat" not in st.session_state: st.session_state.chat = []
chunks, pgs = [], []

if up and model:
    with pdfplumber.open(up) as pdf:
        for i, p in enumerate(pdf.pages):
            txt = p.extract_text()
            if txt:
                for linha in txt.split('\n'):
                    if len(linha.strip()) > 30:
                        chunks.append(linha.strip()); pgs.append(i+1)
    if chunks:
        embs = model.encode(chunks)
        idx = faiss.IndexFlatL2(embs.shape[1])
        idx.add(np.array(embs))

# 5. INTERFACE DE CHAT
st.markdown('<p class="titulo">MentorEdu</p>', unsafe_allow_html=True)

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.write(m["content"])

if prompt := st.chat_input("Pergunte ao Rick..."):
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.write(prompt)

    with st.chat_message("assistant"):
        contexto = ""
        if up and chunks:
            try:
                D, I = idx.search(np.array(model.encode([prompt])), k=2)
                for i in I[0]: contexto += f"[Pág {pgs[i]}] {chunks[i]}\n"
            except: pass

        regras = {
            "Rick Acadêmico": "Você é o Rick Reitor do IFCE. Formal, focado em ABNT.",
            "Rick Sarcástico": "Você é o Rick Sanchez. Sarcástico e genial."
        }

        try:
            msg_final = f"Contexto: {contexto}\n\nPergunta: {prompt}" if contexto else prompt
            res = client.chat.completions.create(
                model="llama-3.1-8
