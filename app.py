import streamlit as st
import pdfplumber, os, faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. CONFIGURAÇÃO VISUAL (ANTI-TEXTO INVISÍVEL)
st.set_page_config(page_title="MentorEdu", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    /* Resolve as caixas brancas dos inputs */
    div[data-baseweb="select"] > div, div[data-testid="stFileUploader"] section {
        background-color: #1c2128 !important;
        border: 1px solid #30363d !important;
    }
    div[data-baseweb="select"] * { color: white !important; }
    /* Estilo das mensagens */
    [data-testid="stChatMessage"] { background-color: #1c2128 !important; border-radius: 10px; }
    .stMarkdown p { color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# 2. MOTOR DE IA
@st.cache_resource
def iniciar():
    try:
        chave = os.getenv("GROQ_API_KEY")
        return Groq(api_key=chave), SentenceTransformer("all-MiniLM-L6-v2")
    except: return None, None

client, model = iniciar()

# 3. BARRA LATERAL (PERSONALIDADES)
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=100)
    st.title("MentorEdu")
    # Removido o Inércia Zero conforme solicitado
    modo = st.selectbox("Personalidade:", ["Rick Acadêmico", "Rick Sarcástico"])
    up = st.file_uploader("Base PDF:", type="pdf")
    if st.button("Limpar"):
        st.session_state.ms = []
        st.rerun()

# 4. MEMÓRIA E PDF
if "ms" not in st.session_state: st.session_state.ms = []
chunks, pgs = [], []

if up and model:
    with pdfplumber.open(up) as pdf:
        for i, p in enumerate(pdf.pages):
            t = p.extract_text()
            if t:
                for linha in t.split('\n'):
                    if len(linha.strip()) > 40:
                        chunks.append(linha.strip())
                        pgs.append(i+1)
    if chunks:
        emb = model.encode(chunks)
        idx = faiss.IndexFlatL2(emb.shape[1])
        idx.add(np.array(emb))

# 5. ÁREA DE CHAT
st.markdown("<h1 style='text-align: center; color: #00d4ff;'>MentorEdu</h1>", unsafe_allow_html=True)

for m in st.session_state.ms:
    with st.chat_message(m["role"]):
        st.write(m["content"])

if p := st.chat_input("Diz aí, Morty..."):
    st.session_state.ms.append({"role": "user", "content": p})
    with st.chat_message("user"): st.write(p)

    with st.chat_message("assistant"):
        ctx = ""
        if up and chunks:
            D, I = idx.search(np.array(model.encode([p])), k=2)
            for i in I[0]: ctx += f"(Pág {pgs[i]}) {chunks[i]} \n\n"

        prompts = {
            "Rick Acadêmico": "Você é o Rick Reitor do IFCE. Focado em ABNT e ciência.",
            "Rick Sarcástico": "Você é o Rick Sanchez. Sarcástico e genial."
        }

        try:
            texto = f"Contexto: {ctx} \n\n Pergunta: {p}" if ctx else p
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "system", "content": prompts[modo]}, {"role": "user", "content": texto}]
            )
            resp = res.choices[0].message.content
            st.write(resp)
            st.session_state.ms.append({"role": "assistant", "content": resp})
        except:
            st.error("Erro na Groq! Verifique a API Key nos Secrets.")
