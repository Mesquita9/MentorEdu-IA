import streamlit as st
import pdfplumber
import os
import numpy as np
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. Configuração da Página (Título que aparece na aba do navegador)
st.set_page_config(page_title="MentorEdu - IFCE", page_icon="🎓", layout="wide")

# 2. CSS para centralizar e estilizar o nome MentorEdu
st.markdown("""
    <style>
    .titulo-principal {
        text-align: center;
        color: #2f9e41; /* Verde do tema que você configurou */
        font-family: 'sans serif';
        font-weight: 800;
        font-size: 3.5rem;
        margin-bottom: -10px;
    }
    .sub-titulo {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Carregamento de Recursos ---
@st.cache_resource
def carregar_ia():
    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key) if api_key else None
    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    return client, modelo

client, modelo_embeddings = carregar_ia()

if not client:
    st.error("Erro: Verifique a GROQ_API_KEY nas configurações do Streamlit Cloud.")
    st.stop()

# --- BARRA LATERAL (Logo e Upload) ---
with st.sidebar:
    # Carrega sua logo.png que está no GitHub
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    
    st.markdown("---")
    st.header("📚 Documentação")
    uploaded_file = st.file_uploader("Carregar PDF para análise", type="pdf")
    
    if st.button("🗑️ Limpar Chat"):
        st.session_state.mensagens = []
        st.rerun()

# --- Processamento RAG (PDF) ---
chunks, paginas = [], []
if uploaded_file:
    with st.spinner("Lendo documento..."):
        with pdfplumber.open(uploaded_file) as pdf:
            for i, page in enumerate(pdf.pages):
                texto = page.extract_text()
                if texto:
                    for linha in texto.split('\n'):
                        if len(linha.strip()) > 50:
                            chunks.append(linha.strip())
                            paginas.append(i + 1)
        
        if chunks:
            embeddings = modelo_embeddings.encode(chunks)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings))

# --- CORPO DO CHAT ---
st.markdown('<h1 class="titulo-principal">MentorEdu</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-titulo">IA Acadêmica de Apoio ao IFCE</p>', unsafe_allow_html=True)

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

# Exibir histórico com interface moderna
for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrada de texto (Chat Input)
if prompt := st.chat_input("Como posso ajudar na sua pesquisa?"):
    st.session_state.mensagens.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        contexto = ""
        if uploaded_file and chunks:
            q_emb = modelo_embeddings.encode([prompt])
            D, I = index.search(np.array(q_emb), k=3)
            for idx in I[0]:
                contexto += f"[Pág {paginas[idx]}] {chunks[idx]}\n"

        try:
            full_prompt = f"Contexto: {contexto}\n\nPergunta: {prompt}" if contexto else prompt
            chat_completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Você é a MentorEdu, uma IA acadêmica desenvolvida para o IFCE."},
                    {"role": "user", "content": full_prompt}
                ]
            )
            resposta = chat_completion.choices[0].message.content
            st.markdown(resposta)
            st.session_state.mensagens.append({"role": "assistant", "content": resposta})
        except Exception as e:
            st.error("Erro na comunicação com a MentorEdu.")
