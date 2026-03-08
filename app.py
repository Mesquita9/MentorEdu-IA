import streamlit as st
from pypdf import PdfReader
import os, faiss, numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. CONFIGURAÇÕES TÉCNICAS E VISUAIS
st.set_page_config(page_title="MentorEdu | IF", page_icon="🎓", layout="wide")
LOGO_IMG = "logo.png"

st.markdown("""
<style>
    .stApp { background-color: #ffffff; }
    .main-title { text-align: center; color: #32a041; font-weight: 800; font-size: 2.5rem; }
    [data-testid="stSidebar"] { border-right: 1px solid #eee; }
    .stButton>button { background-color: #32a041 !important; color: white !important; width: 100%; }
</style>
""", unsafe_allow_html=True)

# 2. CARREGAMENTO DOS MOTORES (CACHE PARA VELOCIDADE)
@st.cache_resource
def iniciar_motores():
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        return Groq(api_key=api_key), SentenceTransformer("all-MiniLM-L6-v2")
    except:
        return None, SentenceTransformer("all-MiniLM-L6-v2")

client, model = iniciar_motores()

# 3. GERENCIAMENTO DE ESTADO (SEM ANINHAMENTO)
if "chat" not in st.session_state:
    st.session_state.chat = []
if "db" not in st.session_state:
    st.session_state.db = None

# 4. INTERFACE LATERAL (LOGO DO IF)
with st.sidebar:
    if os.path.exists(LOGO_IMG):
        st.image(LOGO_IMG, use_container_width=True)
    st.markdown("---")
    area = st.selectbox("Área:", ["Geral", "Informática", "Mecânica", "Eletrotécnica"])
    pdf = st.file_uploader("PDF Didático", type="pdf")
    if st.button("Limpar Sessão"):
        st.session_state.chat = []
        st.session_state.db = None
        st.rerun()

# 5. PROCESSAMENTO DO PDF
if pdf and st.session_state.db is None:
    with st.spinner("Processando material..."):
        reader = PdfReader(pdf)
        txts, pgs = [], []
        for i, page in enumerate(reader.pages):
            content = page.extract_text()
            if not content: continue
            chunks = [content[j:j+500] for j in range(0, len(content), 500)]
            for c in chunks:
                txts.append(c.strip())
                pgs.append(i+1)
        
        if txts:
            vecs = model.encode(txts)
            idx = faiss.IndexFlatL2(vecs.shape[1])
            idx.add(np.array(vecs))
            st.session_state.db = {"idx": idx, "txts": txts, "pgs": pgs}

# 6. EXIBIÇÃO DO CHAT (AVATAR UNIFICADO)
st.markdown('<h1 class="main-title">MentorEdu</h1>', unsafe_allow_html=True)
avatar_oficial = LOGO_IMG if os.path.exists(LOGO_IMG) else None

for m in st.session_state.chat:
    with st.chat_message(m["role"], avatar=avatar_oficial):
        st.write(m["content"])

# 7. LÓGICA DE RESPOSTA (STREAMING)
prompt = st.chat_input("Dúvida acadêmica...")
if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=avatar_oficial):
        st.write(prompt)

    with st.chat_message("assistant", avatar=avatar_oficial):
        contexto = ""
        if st.session_state.db:
            q_v = model.encode([prompt])
            _, ids = st.session_state.db["idx"].search(np.array(q_v), k=2)
            for idx in ids[0]:
                contexto += f"[Pág {st.session_state.db['pgs'][idx]}] {st.session_state.db['txts'][idx]}\n\n"

        sys_p = f"Você é o MentorEdu, assistente do IF. Área: {area}. Responda de forma profissional e didática."
        
        if client:
            stream = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role":"system","content":sys_p}, {"role":"user","content":f"Contexto: {contexto}\n\nPergunta: {prompt}"}],
                stream=True
            )
            res = st.write_stream(stream)
            st.session_state.chat.append({"role": "assistant", "content": res})
        else:
            st.error("Erro: Verifique a GROQ_API_KEY nos Secrets.")
