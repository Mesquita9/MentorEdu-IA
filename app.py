import streamlit as st
from pypdf import PdfReader
import os, faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. CONFIGURAÇÃO E DESIGN (IDENTIDADE IF)
st.set_page_config(page_title="MentorEdu | IF", page_icon="🎓", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; color: #212529; }
    [data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #e0e0e0; }
    .stButton>button { background-color: #32a041 !important; color: white !important; border-radius: 8px; font-weight: 600; }
    .main-title { text-align: center; color: #32a041; font-weight: 800; font-size: 2.5rem; margin-top: -30px; }
    /* Estilo para deixar o chat mais limpo e rápido */
    .stChatMessage { border-radius: 12px; border: 1px solid #eee !important; box-shadow: none !important; }
    </style>
    """, unsafe_allow_html=True)

# 2. CARREGAMENTO OTIMIZADO DOS MODELOS
@st.cache_resource
def load_models():
    try:
        api_key = os.getenv("GROQ_API_KEY")
        return Groq(api_key=api_key), SentenceTransformer("all-MiniLM-L6-v2")
    except: return None, None

client, model = load_models()

# 3. BARRA LATERAL
with st.sidebar:
    # Usa a logo.png do IF conforme combinado
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    
    st.markdown("---")
    area = st.selectbox("ÁREA:", ["Geral", "Matemática", "Física", "Química", "Biologia"])
    up = st.file_uploader("📂 MATERIAL DIDÁTICO (PDF)", type="pdf")
    
    if st.button("Limpar Histórico"):
        st.session_state.chat_history = []
        st.session_state.index = None
        st.rerun()

# 4. MEMÓRIA DE SESSÃO (O segredo da velocidade)
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "index" not in st.session_state: st.session_state.index = None
if "chunks" not in st.session_state: st.session_state.chunks = []

# Processa o PDF apenas se houver um novo arquivo e o index estiver vazio
if up and st.session_state.index is None:
    with st.spinner("Otimizando material para consulta rápida..."):
        reader = PdfReader(up)
        temp_chunks = []
        temp_pgs = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                # Divide em blocos maiores para reduzir o número de cálculos
                lines = text.split('\n')
                for j in range(0, len(lines), 5): 
                    chunk = " ".join(lines[j:j+5]).strip()
                    if len(chunk) > 50:
                        temp_chunks.append(chunk)
                        temp_pgs.append(i + 1)
        
        if temp_chunks:
            embeddings = model.encode(temp_chunks, show_progress_bar=False)
            idx = faiss.IndexFlatL2(embeddings.shape[1])
            idx.add(np.array(embeddings))
            # Salva na sessão para não reprocessar
            st.session_state.index = idx
            st.session_state.chunks = temp_chunks
            st.session_state.pgs = temp_pgs
            st.success("Material pronto!")

# 5. INTERFACE DE CHAT
st.markdown('<h1 class="main-title">MentorEdu</h1>', unsafe_allow_html=True)

for m in st.session_state.chat_history:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Dúvida sobre o material?"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        contexto = ""
        # Busca ultra rápida usando o index em memória
        if st.session_state.index is not None:
            q_emb = model.encode([prompt])
            D, I = st.session_state.index.search(np.array(q_emb), k=2)
            for idx in I[0]:
                contexto += f"[Pág {st.session_state.pgs[idx]}] {st.session_state.chunks[idx]}\n\n"

        sys_msg = f"Você é o MentorEdu, assistente didático do IF. Área: {area}. Seja breve e técnico."
        
        try:
            full_prompt = f"Use este contexto:\n{contexto}\n\nPergunta: {prompt}" if contexto else prompt
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": full_prompt}]
            )
            ans = res.choices[0].message.content
            st.markdown(ans)
            st.session_state.chat_history.append({"role": "assistant", "content": ans})
        except:
            st.error("Erro na API
