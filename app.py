import streamlit as st
import pdfplumber, os, faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. Configuração e Estilo Dark
st.set_page_config(page_title="Inércia Zero", page_icon="🧪", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    section[data-testid="stSidebar"] { background-color: #161b22 !important; }
    .main-title { text-align: center; color: #88e23b; font-family: sans-serif; font-weight: 800; font-size: 3rem; }
    [data-testid="stChatMessage"] { background-color: #21262d; border-radius: 12px; border: 1px solid #30363d; }
    input { color: white !important; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_all():
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return client, model

client, model = load_all()

# 2. Sidebar
with st.sidebar:
    if os.path.exists("logo.png"): st.image("logo.png", width=150)
    st.title("🧪 Inércia Zero")
    var = st.selectbox("Variante do Rick:", ["Rick Sarcástico", "Rick Acadêmico", "Rick Inércia Zero"])
    up = st.file_uploader("📂 PDF", type="pdf")
    if st.button("Resetar"):
        st.session_state.mensagens = []
        st.rerun()

# 3. Processamento RAG
chunks, pgs = [], []
if up:
    with st.spinner("Lendo..."):
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

# 4. Chat
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
            D, I = index.search(np.array(q_emb), k=3)
            for idx in I[0]: ctx += f"[Pág {pgs[idx]}] {chunks[idx]}\n"

        pers = {
            "Rick Sarcástico": "Você é o Rick Sanchez. Sarcástico e chama o usuário de Morty.",
            "Rick Acadêmico": "Você é o Rick Reitor. Científico, ranzinza e focado no IFCE.",
            "Rick Inércia Zero": "Você quer tirar o Morty da inércia com motivação agressiva."
        }
        
        try:
            full_p = f"Contexto: {ctx}\n\nPergunta: {prompt}" if ctx else prompt
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role":"system","content":pers[var]},{"role":"user","content":full_p}]
            )
            ans = res.choices[0].message.content
            st.markdown(ans)
            st.session_state.mensagens.append({"role": "assistant", "content": ans})
        except: st.error("Erro no portal!")
