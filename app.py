import streamlit as st
from pypdf import PdfReader
import os
import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. INTERFACE INSTITUCIONAL IF
st.set_page_config(page_title="MentorEdu | IF", page_icon="🎓")
LOGO_IMG = "logo.png"

st.markdown("""
<style>
    .stApp { background-color: #ffffff; }
    .main-title { text-align: center; color: #32a041; font-weight: 800; font-size: 2.2rem; }
    .stButton>button { background-color: #32a041 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# 2. INICIALIZAÇÃO DE VARIÁVEIS
if "chat" not in st.session_state: 
    st.session_state.chat = []
if "db" not in st.session_state: 
    st.session_state.db = None

# 3. CARREGAMENTO DOS MOTORES
@st.cache_resource
def carregar_modelo():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = carregar_modelo()

# Inicialização do Groq (sem cache)
try:
    chave = st.secrets.get("GROQ_API_KEY")
    if not chave:
        st.error("GROQ_API_KEY não definida nos Secrets do Streamlit.")
        st.stop()
    client = Groq(api_key=chave)
except Exception as e:
    st.error(f"Erro ao inicializar Groq: {e}")
    st.stop()

# 4. BARRA LATERAL (LOGO DO IF)
with st.sidebar:
    if os.path.exists(LOGO_IMG):
        st.image(LOGO_IMG, use_container_width=True)
    st.markdown("---")
    area = st.selectbox("Área Técnica:", ["Geral", "Informática", "Mecânica", "Química"])
    arquivo = st.file_uploader("Upload PDF", type="pdf")
    if st.button("Limpar Tudo"):
        st.session_state.chat = []
        st.session_state.db = None
        st.rerun()

# 5. PROCESSAMENTO DO PDF
if arquivo and st.session_state.db is None:
    with st.spinner("Indexando material..."):
        try:
            leitor = PdfReader(arquivo)
            textos, pgs = [], []
            for i, pagina in enumerate(leitor.pages):
                txt = pagina.extract_text()
                if txt:
                    blocos = [txt[j:j+600] for j in range(0, len(txt), 600)]
                    for b in blocos:
                        textos.append(b.strip())
                        pgs.append(i+1)
            if textos:
                embs = model.encode(textos)
                idx = faiss.IndexFlatL2(embs.shape[1])
                idx.add(np.array(embs))
                st.session_state.db = {"idx": idx, "textos": textos, "pgs": pgs}
            else:
                st.warning("PDF não possui texto extraível.")
        except Exception as e:
            st.error(f"Erro ao processar PDF: {e}")

# 6. CHAT COM AVATAR DO IF
st.markdown('<h1 class="main-title">MentorEdu</h1>', unsafe_allow_html=True)
av = LOGO_IMG if os.path.exists(LOGO_IMG) else None

for m in st.session_state.chat:
    with st.chat_message(m["role"], avatar=av):
        st.write(m["content"])

if prompt := st.chat_input("Dúvida acadêmica?"):
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=av):
        st.write(prompt)

    with st.chat_message("assistant", avatar=av):
        ctx = ""
        if st.session_state.db:
            try:
                v_q = model.encode([prompt])
                _, ids = st.session_state.db["idx"].search(np.array(v_q), k=2)
                for idx_i in ids[0]:
                    ctx += f"[Pág {st.session_state.db['pgs'][idx_i]}] {st.session_state.db['textos'][idx_i]}\n\n"
            except Exception as e:
                st.warning(f"Erro ao buscar contexto no PDF: {e}")

        sys_msg = f"Você é o MentorEdu, tutor oficial do IF. Área: {area}. Responda de forma técnica e didática."
        try:
            stream = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role":"system","content":sys_msg}, {"role":"user","content":f"Contexto: {ctx}\n\nPergunta: {prompt}"}],
                stream=True
            )
            res = st.write_stream(stream)
            st.session_state.chat.append({"role": "assistant", "content": res})
        except Exception as e:
            st.error(f"Erro na geração de resposta: {e}")
