import streamlit as st
import pdfplumber, os, faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. DESIGN DE ELITE (ALTO CONTRASTE)
st.set_page_config(page_title="MentorEdu", page_icon="🧪", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    .stApp { background-color: #0b1117; color: #f0f2f6; }
    
    /* SIDEBAR TOTALMENTE VISÍVEL */
    [data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }
    [data-testid="stSidebar"] * { color: #ffffff !important; font-weight: bold !important; }
    
    /* CORREÇÃO DAS CAIXAS BRANCAS (INPUTS) */
    div[data-baseweb="select"] > div, div[data-testid="stFileUploader"] section {
        background-color: #1e2530 !important;
        border: 1px solid #3b424b !important;
        color: white !important;
    }

    .title { text-align: center; background: linear-gradient(90deg, #00d4ff, #88e23b);
              -webkit-background-clip: text; -webkit-text-fill-color: transparent;
              font-weight: 800; font-size: 3rem; margin-top: -50px; }

    [data-testid="stChatMessage"] { background-color: #1e2530 !important; border-radius: 12px !important; border: 1px solid #30363d !important; }
    [data-testid="stChatMessage"] p { color: #ffffff !important; }
    
    header, footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_all():
    c = Groq(api_key=os.getenv("GROQ_API_KEY"))
    m = SentenceTransformer("all-MiniLM-L6-v2")
    return c, m

client, model = load_all()

# 2. BARRA LATERAL
with st.sidebar:
    st.markdown("### 🧪 PROJETO INÉRCIA ZERO")
    modo = st.selectbox("PERSONALIDADE:", ["Rick Acadêmico", "Rick Inércia Zero", "Rick Sarcástico"])
    up = st.file_uploader("📂 PDF PARA ANÁLISE", type="pdf")
    if st.button("LIMPAR SISTEMA"):
        st.session_state.mensagens = []
        st.rerun()

# 3. LÓGICA RAG
chunks, pgs = [], []
if up:
    with st.spinner("Rick lendo PDF..."):
        with pdfplumber.open(up) as pdf:
            for i, p in enumerate(pdf.pages):
                t = p.extract_text()
                if t:
                    for l in t.split('\n'):
                        if len(l.strip()) > 50:
                            chunks.append(l.strip()); pgs.append(i+1)
        if chunks:
            embs = model.encode(chunks)
            idx_faiss = faiss.IndexFlatL2(embs.shape[1])
            idx_faiss.add(np.array(embs))

# 4. INTERFACE PRINCIPAL
st.markdown('<h1 class="title">MentorEdu</h1>', unsafe_allow_html=True)
if "mensagens" not in st.session_state: st.session_state.mensagens = []

for m in st.session_state.mensagens:
    with st.chat_message(m["role"]):
        st.markdown(f"**{m['role'].upper()}:** {m['content']}")

if prompt := st.chat_input("Diz aí, Morty..."):
    st.session_state.mensagens.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(f"**MORTY:** {prompt}")

    with st.chat_message("assistant"):
        ctx = ""
        if up and chunks:
            q_emb = model.encode([prompt])
            D, I = idx_faiss.search(np.array(q_emb), k=2)
            for idx in I[0]: ctx += f"[Pág {pgs[idx]}] {chunks[idx]}\n\n"

        p_sys = {
            "Rick Acadêmico": "Você é o Rick Reitor do IFCE. Formal e focado em ABNT.",
            "Rick Inércia Zero": "Você é agressivo. Grite para o Morty estudar!",
            "Rick Sarcástico": "Você é o Rick Sanchez clássico. Sarcástico."
        }
        
        try:
            full = f"Contexto:\n{ctx}\n\nPergunta: {prompt}" if ctx else prompt
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role":"system","content":p_sys[modo]},{"role":"user","content":full}]
            )
            ans = res.choices[0].message.content
            st.markdown(f"**RICK:** {ans}")
            st.session_state.mensagens.append({"role": "assistant", "content": ans})
        except: st.error("Erro no portal!")
