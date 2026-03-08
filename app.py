import streamlit as st
from pypdf import PdfReader
import os, faiss, numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. IDENTIDADE VISUAL INSTITUCIONAL
st.set_page_config(page_title="MentorEdu | IF", page_icon="🎓", layout="wide")

# Caminho da logo única
LOGO_PATH = "logo.png"

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    .main-title { text-align: center; color: #32a041; font-weight: 800; font-size: 2.5rem; margin-top: -40px; }
    [data-testid="stSidebar"] { border-right: 1px solid #eee; }
    .stButton>button { background-color: #32a041 !important; color: white !important; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# 2. CARREGAMENTO DOS MOTORES
@st.cache_resource
def load_ia():
    try:
        key = st.secrets["GROQ_API_KEY"]
        return Groq(api_key=key), SentenceTransformer("all-MiniLM-L6-v2")
    except:
        return None, SentenceTransformer("all-MiniLM-L6-v2")

client, model = load_ia()

if "chat" not in st.session_state: st.session_state.chat = []
if "db" not in st.session_state: st.session_state.db = None

# 3. BARRA LATERAL
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    
    st.markdown("---")
    area = st.selectbox("Área Técnica:", ["Geral", "Informática", "Mecânica", "Eletrotécnica", "Química"])
    pdf_file = st.file_uploader("Subir Material (PDF)", type="pdf")
    
    if st.button("Limpar Histórico"):
        st.session_state.chat = []
        st.session_state.db = None
        st.rerun()

# 4. PROCESSAMENTO DO PDF (CACHE)
if pdf_file and st.session_state.db is None:
    with st.spinner("Preparando base de conhecimento..."):
        reader = PdfReader(pdf_file)
        texts, pgs = [], []
        for i, page in enumerate(reader.pages):
            content = page.extract_text()
            if content:
                # Divide em blocos para busca eficiente
                chunks = [content[j:j+600] for j in range(0, len(content), 600)]
                for c in chunks:
                    if len(c.strip()) > 50:
                        texts.append(c.strip())
                        pgs.append(i+1)
        
        if texts:
            embeddings = model.encode(texts)
            idx = faiss.IndexFlatL2(embeddings.shape[1])
            idx.add(np.array(embeddings))
            st.session_state.db = {"idx": idx, "texts": texts, "pgs": pgs}

# 5. CHAT COM AVATARES UNIFICADOS (IF)
st.markdown('<h1 class="main-title">MentorEdu</h1>', unsafe_allow_html=True)

# Define a logo como avatar para ambos
avatar_img = LOGO_PATH if os.path.exists(LOGO_PATH) else None

for m in st.session_state.chat:
    with st.chat_message(m["role"], avatar=avatar_img):
        st.write(m["content"])

if prompt := st.chat_input("Tire sua dúvida acadêmica..."):
    # Mensagem do Aluno
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=avatar_img):
        st.write(prompt)

    # Resposta do Mentor (Professor)
    with st.chat_message("assistant", avatar=avatar_img):
        contexto = ""
        if st.session_state.db:
            q_emb = model.encode([prompt])
            _, ids = st.session_state.db["idx"].search(np.array(q_emb), k=2)
            for idx in ids[0]:
                contexto += f"[Pág {st.session_state.db['pgs'][idx]}] {st.session_state.db['texts'][idx]}\n\n"

        if client:
            sys_msg = f"Você é o MentorEdu, assistente do IF. Área: {area}. Responda de forma técnica e clara."
            stream = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": f"Contexto:\n{contexto}\n\nPergunta: {prompt}"}
                ],
                stream=True
            )
            res_txt = st.write_stream(stream)
            st.session_state.chat.append({"role": "assistant", "content": res_txt})
        else:
            st.error("GROQ_API_KEY não configurada nos Secrets.")
