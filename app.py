import streamlit as st
from pypdf import PdfReader
import os, faiss, numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. CONFIGURAÇÃO VISUAL (LOGO E CORES DO IF)
st.set_page_config(page_title="MentorEdu | IF", page_icon="🎓")

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    .main-title { text-align: center; color: #32a041; font-weight: 800; font-size: 2.5rem; }
    .stButton>button { background-color: #32a041 !important; color: white !important; width: 100%; border-radius: 8px; }
    [data-testid="stSidebar"] { border-right: 1px solid #eee; }
    </style>
    """, unsafe_allow_html=True)

# 2. CARREGAMENTO DOS MOTORES (CACHE)
@st.cache_resource
def iniciar_ia():
    api_key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=api_key), SentenceTransformer("all-MiniLM-L6-v2")

client, model = iniciar_ia()

# 3. BARRA LATERAL COM LOGO.PNG
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    st.markdown("---")
    area = st.selectbox("Área de Estudo:", ["Geral", "Matemática", "Física", "Química"])
    pdf_file = st.file_uploader("Subir PDF Didático", type="pdf")
    if st.button("Limpar Conversa"):
        st.session_state.clear()
        st.rerun()

# 4. GESTÃO DE MEMÓRIA (EVITA LENTIDÃO)
if "chat" not in st.session_state: st.session_state.chat = []
if "db" not in st.session_state: st.session_state.db = None

# Processamento simplificado do PDF
if pdf_file and st.session_state.db is None:
    with st.spinner("Lendo material..."):
        reader = PdfReader(pdf_file)
        texts, pgs = [], []
        for i, page in enumerate(reader.pages):
            content = page.extract_text()
            if content:
                # Divide o texto em blocos fixos (mais rápido e estável)
                parts = [content[x:x+600] for x in range(0, len(content), 600)]
                for p in parts:
                    texts.append(p.strip())
                    pgs.append(i+1)
        
        embeddings = model.encode(texts)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))
        st.session_state.db = {"index": index, "texts": texts, "pgs": pgs}

# 5. CHAT PROFISSIONAL
st.markdown('<h1 class="main-title">MentorEdu</h1>', unsafe_allow_html=True)

for m in st.session_state.chat:
    with st.chat_message(m["role"]): st.write(m["content"])

if prompt := st.chat_input("Como posso ajudar?"):
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.write(prompt)

    with st.chat_message("assistant"):
        ctx = ""
        if st.session_state.db:
            q_vec = model.encode([prompt])
            _, ids = st.session_state.db["index"].search(np.array(q_vec), k=2)
            for idx in ids[0]:
                ctx += f"[Pág {st.session_state.db['pgs'][idx]}] {st.session_state.db['texts'][idx]}\n\n"

        sys_p = f"Você é o MentorEdu, assistente do IF na área de {area}. Seja técnico e cordial."
        
        # Streaming para eliminar a sensação de lentidão
        stream = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": sys_p}, 
                      {"role": "user", "content": f"Contexto:\n{ctx}\n\nPergunta: {prompt}"}],
            stream=True
        )
        full_res = st.write_stream(stream)
        st.session_state.chat.append({"role": "assistant", "content": full_res})
