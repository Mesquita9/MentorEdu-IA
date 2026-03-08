import streamlit as st
import pdfplumber, os, faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# --- CONFIGURAÇÃO E ESTILO DARK (GEMINI STYLE) ---
st.set_page_config(page_title="Inércia Zero", page_icon="🧪", layout="wide")

st.markdown("""
    <style>
    /* Fundo Escuro e Texto Claro */
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    [data-testid="stSidebar"] { background-color: #161b22 !important; }
    
    /* Título MentorEdu */
    .main-title { text-align: center; color: #88e23b; font-weight: 800; font-size: 3rem; margin-top: -60px; }
    
    /* Balões de Chat Estilizados */
    [data-testid="stChatMessage"] { background-color: #161b22 !important; border: 1px solid #30363d !important; border-radius: 12px; }
    
    /* Barra de Chat Escura e Fixa */
    .stChatInputContainer { background-color: #0e1117 !important; padding-bottom: 20px; }
    .stChatInputContainer div { background-color: #21262d !important; border: 1px solid #444c56 !important; }
    textarea { color: white !important; }
    
    /* Remover elementos padrão do Streamlit */
    header, footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def init_models():
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return client, model

client, model = init_models()

# --- BARRA LATERAL ---
with st.sidebar:
    if os.path.exists("logo.png"): st.image("logo.png", width=120)
    st.title("🧪 Painel de Controle")
    variante = st.selectbox("Variante do Rick:", ["Rick Sarcástico", "Rick Acadêmico", "Rick Inércia Zero"])
    pdf_file = st.file_uploader("📂 Subir PDF", type="pdf")
    if st.button("Limpar Histórico"):
        st.session_state.mensagens = []
        st.rerun()

# --- PROCESSAMENTO DO PDF (RAG) ---
chunks, pgs = [], []
if pdf_file:
    with st.spinner("Rick está lendo..."):
        with pdfplumber.open(pdf_file) as pdf:
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

# --- INTERFACE DE CHAT ---
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
        if pdf_file and chunks:
            q_emb = model.encode([prompt])
            D, I = index.search(np.array(q_emb), k=2)
            for idx in I[0]: ctx += f"[Pág {pgs[idx]}] {chunks[idx]}\n "

        pers = {
            "Rick Sarcástico": "Você é o Rick Sanchez. Sarcástico e genial.",
            "Rick Acadêmico": "Você é o Rick Reitor do IFCE. Científico e ranzinza.",
            "Rick Inércia Zero": "Você quer tirar o Morty da inércia com motivação agressiva."
        }
        
        try:
            full_p = f"Contexto: {ctx}\n\nPergunta: {prompt}" if ctx else prompt
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role":"system","content":pers[variante]},{"role":"user","content":full_p}]
            )
            ans = res.choices[0].message.content
            st.markdown(ans)
            st.session_state.mensagens.append({"role": "assistant", "content": ans})
        except: st.error("O portal de IA falhou, Morty!")
