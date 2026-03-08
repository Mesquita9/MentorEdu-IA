import streamlit as st
from pypdf import PdfReader
import os, faiss, time
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. INTERFACE INSTITUCIONAL OTIMIZADA
st.set_page_config(page_title="MentorEdu | IF", page_icon="🎓", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #fcfcfc; color: #1a1a1a; }
    [data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #eee; }
    
    /* Cores IF */
    .stButton>button { 
        background-color: #32a041 !important; 
        color: white !important; 
        border-radius: 6px; 
        transition: 0.2s;
    }
    .main-title { 
        text-align: center; color: #32a041; 
        font-weight: 800; font-size: 2.2rem; margin-top: -40px; 
    }
    /* Chat bubbles mais leves */
    [data-testid="stChatMessage"] { 
        background-color: #ffffff !important; 
        border: 1px solid #f0f0f0 !important; 
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. CARREGAMENTO "LAZY" (SÓ CARREGA UMA VEZ)
@st.cache_resource
def get_ia_engines():
    key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=key), SentenceTransformer("all-MiniLM-L6-v2")

client, model = get_ia_engines()

# 3. BARRA LATERAL
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    
    st.markdown("---")
    area = st.selectbox("Área Técnica:", ["Geral", "Informática", "Mecânica", "Eletrotécnica", "Administração"])
    uploaded_file = st.file_uploader("📂 Base de Conhecimento (PDF)", type="pdf")
    
    if st.button("🔄 Reiniciar Sessão"):
        st.session_state.clear()
        st.rerun()

# 4. PROCESSAMENTO DE ALTA VELOCIDADE
if "history" not in st.session_state: st.session_state.history = []
if "vector_db" not in st.session_state: st.session_state.vector_db = None

if uploaded_file and st.session_state.vector_db is None:
    with st.status("Preparando material didático...") as status:
        reader = PdfReader(uploaded_file)
        text_chunks, pgs = [], []
        
        for i, page in enumerate(reader.pages):
            content = page.extract_text()
            if content:
                # Blocos de 600 caracteres para equilíbrio entre precisão e rapidez
                for start in range(0, len(content), 600):
                    chunk = content[start:start+650].strip()
                    if len(chunk) > 100:
                        text_chunks.append(chunk)
                        pgs.append(i + 1)
        
        if text_chunks:
            embs = model.encode(text_chunks, show_progress_bar=False)
            idx = faiss.IndexFlatL2(embs.shape[1])
            idx.add(np.array(embs))
            
            st.session_state.vector_db = {"idx": idx, "chunks": text_chunks, "pgs": pgs}
            status.update(label="Material pronto para consulta!", state="complete")

# 5. ÁREA DE CHAT (STREAMING)
st.markdown('<h1 class="main-title">MentorEdu</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Assistente de Apoio ao Estudante</p>", unsafe_allow_html=True)

for msg in st.session_state.history:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Como posso ajudar nos seus estudos?"):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        context = ""
        # Busca no PDF em microssegundos
        if st.session_state.vector_db:
            q_emb = model.encode([prompt])
            D, I = st.session_state.vector_db["idx"].search(np.array(q_emb), k=2)
            for idx in I[0]:
                context += f"[Pág {st.session_state.vector_db['pgs'][idx]}] {st.session_state.vector_db['chunks'][idx]}\n\n"

        sys_msg = f"Você é o MentorEdu, assistente didático do IF. Área: {area}. Seja direto, técnico e encorajador."
        
        try:
            full_input = f"CONTEXTO DO MATERIAL:\n{context}\n\nPERGUNTA DO ALUNO: {prompt}" if context else prompt
            
            # STREAMING: A resposta aparece enquanto é gerada
            stream = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": full_input}],
                stream=True
            )
            
            response = st.write_stream(stream)
            st.session_state.history.append({"role": "assistant", "content": response})
        except:
            st.error("Conexão instável. Tente novamente em
