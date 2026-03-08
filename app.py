import streamlit as st
import pdfplumber, os, faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. TEMA E CSS (CORREÇÃO DAS CAIXAS BRANCAS)
st.set_page_config(page_title="MentorEdu", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    /* Resolve as caixas brancas invisíveis */
    div[data-baseweb="select"] > div, div[data-testid="stFileUploader"] section {
        background-color: #1c2128 !important;
        border: 1px solid #30363d !important;
        color: white !important;
    }
    div[data-baseweb="select"] span, label p { color: white !important; }
    /* Ajuste das mensagens de chat */
    [data-testid="stChatMessage"] { background-color: #1c2128 !important; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# 2. INICIALIZAÇÃO DA IA
@st.cache_resource
def load_models():
    try:
        key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        return Groq(api_key=key), SentenceTransformer("all-MiniLM-L6-v2")
    except: return None, None

client, model = load_models()

# 3. BARRA LATERAL (CONFIGURAÇÃO)
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=120)
    st.title("MentorEdu")
    
    # Personalidades conforme sua regra (Inércia Zero removido)
    modo = st.selectbox("QUEM É O RICK?", ["Rick Acadêmico", "Rick Sarcástico"])
    up = st.file_uploader("📂 BASE PDF", type="pdf")
    
    if st.button("LIMPAR CONVERSA"):
        st.session_state.chat = []
        st.rerun()

# 4. LÓGICA DE MEMÓRIA E PDF
if "chat" not in st.session_state: st.session_state.chat = []
chunks, pgs = [], []

if up and model:
    try:
        with pdfplumber.open(up) as pdf:
            for i, p in enumerate(pdf.pages):
                txt = p.extract_text()
                if txt:
                    for lin in txt.split('\n'):
                        if len(lin.strip()) > 40:
                            chunks.append(lin.strip())
                            pgs.append(i+1)
        if chunks:
            embs = model.encode(chunks)
            idx = faiss.IndexFlatL2(embs.shape[1])
            idx.add(np.array(embs))
    except: st.error("Erro ao ler o arquivo PDF.")

# 5. INTERFACE DE CHAT
st.markdown("<h1 style='text-align: center; color: #00d4ff;'>MentorEdu</h1>", unsafe_allow_html=True)

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.write(m["content"])

if prompt := st.chat_input("Diz aí, Morty..."):
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        contexto = ""
        if up and chunks:
            try:
                q_emb = model.encode([prompt])
                D, I = idx.search(np.array(q_emb), k=2)
                for i in I[0]: contexto += f"(Pág {pgs[i]}) {chunks[i]}\n\n"
            except: pass

        p_sys = {
            "Rick Acadêmico": "Você é o Rick Reitor do IFCE. Formal, focado em ABNT e ciência.",
            "Rick Sarcástico": "Você é o Rick Sanchez. Sarcástico, genial e impaciente."
        }

        try:
            final_p = f"Contexto: {contexto}\n\nPergunta: {prompt}" if contexto else prompt
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "system", "content": p_sys[modo]}, {"role": "user", "content": final_p}]
            )
            resp = res.choices[0].message.content
            st.write(resp)
            st.session_state.chat.append({"role": "assistant", "content": resp})
        except:
            st.error("Erro na API Groq! Verifique sua chave.")
