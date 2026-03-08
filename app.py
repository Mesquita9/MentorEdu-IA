import streamlit as st
from pypdf import PdfReader
import os, faiss, numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. CONFIGURAÇÃO VISUAL INSTITUCIONAL (IF)
st.set_page_config(page_title="MentorEdu | IF", page_icon="🎓")

st.markdown("""
<style>
    .stApp { background-color: #ffffff; }
    .main-title { text-align: center; color: #32a041; font-weight: 800; font-size: 2.5rem; }
    .stButton>button { background-color: #32a041 !important; color: white !important; width: 100%; border-radius: 8px; }
    [data-testid="stSidebar"] { border-right: 1px solid #eee; }
</style>
""", unsafe_allow_html=True)

# 2. INICIALIZAÇÃO DE MOTORES E ESTADO (CACHE)
@st.cache_resource
def iniciar_motores():
    key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=key), SentenceTransformer("all-MiniLM-L6-v2")

client, model = iniciar_motores()

if "chat" not in st.session_state: st.session_state.chat = []
if "db" not in st.session_state: st.session_state.db = None

# 3. BARRA LATERAL COM LOGO DO IF
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    st.markdown("---")
    area = st.selectbox("Área de Estudo:", ["Geral", "Matemática", "Física", "Química"])
    pdf_file = st.file_uploader("Subir Material (PDF)", type="pdf")
    if st.button("Limpar Histórico"):
        st.session_state.chat = []
        st.session_state.db = None
        st.rerun()

# 4. PROCESSAMENTO DO MATERIAL (SÓ RODA SE O DB ESTIVER VAZIO)
if pdf_file and st.session_state.db is None:
    with st.spinner("Analisando material didático..."):
        reader = PdfReader(pdf_file)
        texts, pgs = [], []
        for i, page in enumerate(reader.pages):
            content = page.extract_text()
            if content:
                # Divide o texto em blocos para busca rápida
                parts = [content[x:x+500] for x in range(0, len(content), 500)]
                for p in parts:
                    if len(p.strip()) > 30:
                        texts.append(p.strip())
                        pgs.append(i+1)
        
        if texts:
            embs = model.encode(texts)
            idx = faiss.IndexFlatL2(embs.shape[1])
            idx.add(np.array(embs))
            st.session_state.db = {"index": idx, "texts": texts, "pgs": pgs}

# 5. INTERFACE DE CHAT E LÓGICA DE RESPOSTA
st.markdown('<h1 class="main-title">MentorEdu</h1>', unsafe_allow_html=True)

for m in st.session_state.chat:
    with st.chat_message(m["role"]): st.write(m["content"])

if prompt := st.chat_input("Tire sua dúvida aqui..."):
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.write(prompt)

    with st.chat_message("assistant"):
        ctx = ""
        if st.session_state.db:
            q_vec = model.encode([prompt])
            _, ids = st.session_state.db["index"].search(np.array(q_vec), k=2)
            for i in ids[0]:
                ctx += f"[Pág {st.session_state.db['pgs'][i]}] {st.session_state.db['texts'][i]}\n\n"

        sys_msg = f"Você é o MentorEdu, assistente didático do IF. Área: {area}. Seja profissional e direto."
        
        stream = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": sys_msg}, 
                      {"role": "user", "content": f"Contexto:\n{ctx}\n\nPergunta: {prompt}"}],
            stream=True
        )
        res_completa = st.write_stream(stream)
        st.session_state.chat.append({"role": "assistant", "content": res_completa})
