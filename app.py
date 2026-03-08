import streamlit as st
import pdfplumber
import os
import numpy as np
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. Configuração da Página
st.set_page_config(page_title="Inércia Zero - MentorEdu", page_icon="🧪", layout="wide")

# 2. Estilização Rick and Morty / Inércia Zero
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #97ce4c; /* Verde Portal */
        font-family: 'Courier New', Courier, monospace;
        font-weight: 900;
        font-size: 3.5rem;
        text-shadow: 2px 2px #44281d;
    }
    .subtitle {
        text-align: center;
        color: #88e23b;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    /* Estilo para as mensagens */
    .stChatMessage {
        border-radius: 15px;
        border: 2px solid #97ce4c;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Inicialização ---
@st.cache_resource
def load_resources():
    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key) if api_key else None
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return client, model

client, model = load_resources()

# --- BARRA LATERAL (Portal de Comando) ---
with st.sidebar:
    st.image("logo.png", width=150) # Rick na sidebar
    st.markdown("### 🧪 Projeto Inércia Zero")
    st.info("Morty, coloca o PDF aqui ou a gente nunca vai sair dessa dimensão acadêmica!")
    uploaded_file = st.file_uploader("Subir PDF", type="pdf", label_visibility="collapsed")
    
    if st.button("Explodir Histórico (Reset)"):
        st.session_state.mensagens = []
        st.rerun()

# --- Processamento RAG (Cérebro do Rick) ---
chunks, paginas = [], []
if uploaded_file:
    with st.spinner("Analisando... isso é ciência de verdade, Morty!"):
        with pdfplumber.open(uploaded_file) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    for line in text.split('\n'):
                        if len(line.strip()) > 50:
                            chunks.append(line.strip())
                            paginas.append(i + 1)
        if chunks:
            embeddings = model.encode(chunks)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings))

# --- CORPO DO CHAT ---
st.markdown('<h1 class="main-title">MentorEdu</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Dimensão: Projeto Inércia Zero</p>', unsafe_allow_html=True)

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

# Exibição das mensagens com as imagens que você subiu
for msg in st.session_state.mensagens:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="logo2.png"): # Morty
            st.markdown(f"**Morty:** {msg['content']}")
    else:
        with st.chat_message("assistant", avatar="logo.png"): # Rick
            st.markdown(f"**Rick:** {msg['content']}")

# Input de Chat
if prompt := st.chat_input("Fala logo, Morty..."):
    st.session_state.mensagens.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="logo2.png"):
        st.markdown(f"**Morty:** {prompt}")

    with st.chat_message("assistant", avatar="logo.png"):
        contexto = ""
        if uploaded_file and chunks:
            q_emb = model.encode([prompt])
            D, I = index.search(np.array(q_emb), k=3)
            for idx in I[0]:
                contexto += f"[Página {paginas[idx]}] {chunks[idx]}\n"

        try:
            full_prompt = f"Contexto do PDF:\n{contexto}\n\nPergunta do Morty: {prompt}" if contexto else prompt
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": """
                    Você é o Rick Sanchez. Você é o mentor do 'Projeto Inércia Zero'.
                    Sua personalidade: Gênio, sarcástico, impaciente, usa 'Wubba Lubba Dub Dub' e chama o usuário de Morty.
                    Se houver contexto de PDF, use-o para dar uma resposta cientificamente perfeita, mas com seu jeito grosso.
                    Se não houver PDF, apenas responda como o Rick faria.
                    """},
                    {"role": "user", "content": full_prompt}
                ]
            )
            answer = response.choices[0].message.content
            st.markdown(f"**Rick:** {answer}")
            st.session_state.mensagens.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error("O portal deu erro, Morty! A culpa é sua!")
