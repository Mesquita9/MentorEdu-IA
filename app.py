import streamlit as st
from pypdf import PdfReader
import os, faiss, numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. CONFIGURAÇÃO DE INTERFACE
st.set_page_config(page_title="MentorEdu | IF", page_icon="🎓")
LOGO_IMG = "logo.png"

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    .main-title { text-align: center; color: #32a041; font-weight: 800; font-size: 2.2rem; }
    .stButton>button { background-color: #32a041 !important; color: white !important; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# 2. CARREGAMENTO DOS COMPONENTES (CACHE)
@st.cache_resource
def inicializar_motores():
    try:
        chave = st.secrets["GROQ_API_KEY"]
        return Groq(api_key=chave), SentenceTransformer("all-MiniLM-L6-v2")
    except:
        return None, SentenceTransformer("all-MiniLM-L6-v2")

client, model = inicializar_motores()

# Inicializa estados de memória para evitar erros de variável inexistente
if "chat" not in st.session_state: st.session_state.chat = []
if "db" not in st.session_state: st.session_state.db = None

# 3. BARRA LATERAL (IDENTIDADE IF)
with st.sidebar:
    if os.path.exists(LOGO_IMG):
        st.image(LOGO_IMG, use_container_width=True)
    st.markdown("---")
    area = st.selectbox("Área Técnica:", ["Geral", "Informática", "Mecânica", "Química"])
    arquivo = st.file_uploader("Upload do Material (PDF)", type="pdf")
    if st.button("Limpar Tudo"):
        st.session_state.chat = []
        st.session_state.db = None
        st.rerun()

# 4. PROCESSAMENTO DO PDF
if arquivo and st.session_state.db is None:
    with st.spinner("Indexando..."):
        leitor = PdfReader(arquivo)
        textos, pgs = [], []
        for i, pagina in enumerate(leitor.pages):
            texto = pagina.extract_text()
            if texto:
                blocos = [texto[j:j+600] for j in range(0, len(texto), 600)]
                for b in blocos:
                    textos.append(b.strip())
                    pgs.append(i+1)
        if textos:
            embs = model.encode(textos)
            idx = faiss.IndexFlatL2(embs.shape[1])
            idx.add(np.array(embs))
            st.session_state.db = {"idx": idx, "textos": textos, "pgs": pgs}

# 5. CHAT COM AVATARES UNIFICADOS (IF)
st.markdown('<h1 class="main-title">MentorEdu</h1>', unsafe_allow_html=True)

# Avatar institucional para ambos os lados
av = LOGO_IMG if os.path.exists(LOGO_IMG) else None

for m in st.session_state.chat:
    with st.chat_message(m["role"], avatar=av):
        st.write(m["content"])

if prompt := st.chat_input("Dúvida acadêmica?"):
    # Mensagem do Aluno
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=av):
        st.write(prompt)

    # Resposta do Mentor (Streaming)
    with st.chat_message("assistant", avatar=av):
        contexto = ""
        if st.session_state.db:
            v_q = model.encode([prompt])
            _, ids = st.session_state.db["idx"].search(np.array(v_q), k=2)
            for id_idx in ids[0]:
                contexto += f"[Pág {st.session_state.db['pgs'][id_idx]}] {st.session_state.db['textos'][id_idx]}\n\n"

        if client:
            sys = f"Você é o MentorEdu, assistente oficial do IF. Área: {area}. Responda de forma técnica."
            fluxo = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role":"system","content":sys}, {"role":"user","content":f"Contexto: {contexto}\n\nPergunta: {prompt}"}],
                stream=True
            )
            res = st.write_stream(fluxo)
            st.session_state.chat.append({"role": "assistant", "content": res})
        else:
            st.error("GROQ_API_KEY ausente nos Secrets.")
