import streamlit as st
import pdfplumber, os, faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. DESIGN CORRIGIDO (ALTO CONTRASTE E VISIBILIDADE)
st.set_page_config(page_title="MentorEdu", page_icon="🧪", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    .stApp { background-color: #0e1117; color: #ffffff; font-family: 'Inter', sans-serif; }

    /* SIDEBAR - TEXTO BRANCO E FUNDO ESCURO */
    [data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
        color: #ffffff !important; font-weight: 700 !important;
    }

    /* CORREÇÃO DAS CAIXAS BRANCAS (SELECTBOX E UPLOADER) */
    div[data-baseweb="select"] > div, div[data-testid="stFileUploader"] section {
        background-color: #1e2530 !important;
        border: 2px solid #3b424b !important;
        color: white !important;
    }
    
    .main-title { 
        text-align: center; background: linear-gradient(90deg, #00d4ff, #88e23b);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 3rem; margin-top: -40px; 
    }

    [data-testid="stChatMessage"] {
        background-color: #1c2128 !important; border: 1px solid #30363d !important; border-radius: 12px;
    }
    header, footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    try:
        c = Groq(api_key=os.getenv("GROQ_API_KEY"))
        m = SentenceTransformer("all-MiniLM-L6-v2")
        return c, m
    except: return None, None

client, model = load_engine()

# 2. BARRA LATERAL (PERSONALIDADES ATUALIZADAS)
with st.sidebar:
    st.markdown("## 🧪 MENTOREDU")
    st.markdown("---")
    # Removido Inércia Zero, mantidos Académico e Sarcástico
    modo = st.selectbox("ESTILO DO RICK:", ["Rick Académico", "Rick Sarcástico"])
    up = st.file_uploader("📂 SUBIR BASE (PDF)", type="pdf")
    if st.button("LIMPAR HISTÓRICO"):
        st.session_state.mensagens = []
        st.rerun()

# 3. LÓGICA RAG
if "mensagens" not in st.session_state: st.session_state.mensagens = []
chunks, pgs = [], []

if up and model:
    with st.spinner("Rick a analisar o PDF..."):
        with pdfplumber.open(up) as pdf:
            for i, p in enumerate(pdf.pages):
                txt = p.extract_text()
                if txt:
                    for line in txt.split('\n'):
                        if len(line.strip()) > 40:
                            chunks.append(line.strip()); pgs.append(i+1)
        if chunks:
            embs = model.encode(chunks)
            idx = faiss.IndexFlatL2(embs.shape[1])
            idx.add(np.array(embs))

# 4. INTERFACE DE CHAT
st.markdown('<h1 class="main-title">MentorEdu</h1>', unsafe_allow_html=True)

for m in st.session_state.mensagens:
    with st.chat_message(m["role"]):
        st.markdown(f"**{m['role'].upper()}:** {m['content']}")

if prompt := st.chat_input("Pergunta ao Rick..."):
    st.session_state.mensagens.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(f"**MORTY:** {prompt}")

    with st.chat_message("assistant"):
        ctx = ""
        if up and chunks:
            q_emb = model.encode([prompt])
            D, I = idx.search(np.array(q_emb), k=2)
            for i in I[0]: ctx += f"[Pág {pgs[i]}] {chunks[i]}\n\n"

        p_sys = {
            "Rick Académico": "Você é o Rick Reitor do IFCE. Focado em normas ABNT e rigor científico.",
            "Rick Sarcástico": "Você é o Rick Sanchez clássico. Sarcástico, brilhante e impaciente."
        }

        try:
            full_p = f"Contexto:\n{ctx}\n\nPergunta: {prompt}" if ctx else prompt
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "system", "content": p_sys[modo]}, {"role": "user", "content": full_p}]
            )
            ans = res.choices[0].message.content
            st.markdown(f"**RICK:** {ans}")
            st.session_state.mensagens.append({"role": "assistant", "content": ans})
        except: st.error("Erro na Groq!")
