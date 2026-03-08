import streamlit as st
import pdfplumber
import os
import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# --- 1. CONFIGURAÇÃO BÁSICA ---
st.set_page_config(page_title="MentorEdu", page_icon="🧪", layout="wide")

# CSS super simplificado para forçar texto branco e fundo escuro
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    [data-testid="stSidebar"] { background-color: #161b22; }
    div[data-testid="stChatMessage"] { background-color: #1c2128; border: 1px solid #30363d; }
    p, span, div { color: white !important; }
</style>
""", unsafe_allow_html=True)

# --- 2. CARREGAR IA ---
@st.cache_resource
def load_engine():
    try:
        # Pega a chave do ambiente (Streamlit Secrets)
        c = Groq(api_key=os.getenv("GROQ_API_KEY"))
        m = SentenceTransformer("all-MiniLM-L6-v2")
        return c, m
    except Exception:
        return None, None

client, model = load_engine()

# --- 3. BARRA LATERAL ---
with st.sidebar:
    st.markdown("## 🧪 MENTOREDU")
    modo = st.selectbox("ESTILO DO RICK:", ["Rick Acadêmico", "Rick Sarcástico"])
    up = st.file_uploader("📂 SUBIR BASE (PDF)", type="pdf")
    
    if st.button("LIMPAR HISTÓRICO"):
        st.session_state.mensagens = []
        st.rerun()

# --- 4. MEMÓRIA E PDF ---
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

chunks = []
pgs = []

if up and model:
    with st.spinner("Lendo PDF..."):
        try:
            with pdfplumber.open(up) as pdf:
                for i, p in enumerate(pdf.pages):
                    txt = p.extract_text()
                    if txt:
                        for line in txt.split('\n'):
                            if len(line.strip()) > 40:
                                chunks.append(line.strip())
                                pgs.append(i + 1)
            
            if chunks:
                embs = model.encode(chunks)
                idx = faiss.IndexFlatL2(embs.shape[1])
                idx.add(np.array(embs))
        except Exception:
            st.error("Erro ao ler o PDF.")

# --- 5. CHAT ---
st.title("MentorEdu")

for m in st.session_state.mensagens:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Diz aí, Morty..."):
    # Adiciona a pergunta na tela
    st.session_state.mensagens.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Processa a resposta
    with st.chat_message("assistant"):
        ctx = ""
        if up and chunks and model:
            try:
                q_emb = model.encode([prompt])
                D, I = idx.search(np.array(q_emb), k=2)
                for i in I[0]:
                    ctx = ctx + "[Pág " + str(pgs[i]) + "] " + chunks[i] + "\n\n"
            except Exception:
                pass

        p_sys = {
            "Rick Acadêmico": "Você é o Rick Reitor do IFCE. Focado em ciência pura e normas.",
            "Rick Sarcástico": "Você é o Rick Sanchez clássico. Sarcástico e impaciente."
        }

        try:
            if ctx:
                texto_final = "Contexto: " + ctx + "\nPergunta: " + prompt
            else:
                texto_final = prompt

            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": p_sys[modo]},
                    {"role": "user", "content": texto_final}
                ]
            )
            ans = res.choices[0].message.content
            
            st.markdown(ans)
            st.session_state.mensagens.append({"role": "assistant", "content": ans})
        except Exception:
            st.error("Erro na API da Groq. Verifique sua chave.")
