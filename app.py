import streamlit as st
import pdfplumber
import os
import numpy as np
import faiss

from groq import Groq
from sentence_transformers import SentenceTransformer

st.title("IA que conversa com PDF")

# verificar API
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY não encontrada nas Secrets.")
    st.stop()

client = Groq(api_key=api_key)

# modelo de embeddings
modelo_embeddings = SentenceTransformer("all-MiniLM-L6-v2")

# upload do PDF
uploaded_file = st.file_uploader("Envie um PDF", type="pdf")

if uploaded_file:

    chunks = []
    paginas = []

    with pdfplumber.open(uploaded_file) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                partes = [page_text[j:j+500] for j in range(0, len(page_text), 500)]
                for p in partes:
                    chunks.append(p)
                    paginas.append(i + 1)

    if len(chunks) == 0:
        st.error("Não foi possível extrair texto do PDF.")
        st.stop()

    # criar embeddings
    embeddings = modelo_embeddings.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # iniciar sessão de chat
    if "mensagens" not in st.session_state:
        st.session_state.mensagens = []

    pergunta = st.text_input("Faça uma pergunta sobre o PDF")

    if pergunta:
        # salvar pergunta no histórico
        st.session_state.mensagens.append({"role": "user", "content": pergunta})

        try:
            pergunta_embedding = modelo_embeddings.encode([pergunta])
            D, I = index.search(np.array(pergunta_embedding), k=3)  # k=3

            contexto = ""
            for i in I[0]:
                contexto += f"(Página {paginas[i]}) {chunks[i]}\n"

            contexto = contexto[:3000]

            prompt = f"""
Use apenas o contexto abaixo para responder.

Contexto:
{contexto}

Pergunta:
{pergunta}
"""

            resposta = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=st.session_state.mensagens,
                max_tokens=400
            )

            conteudo_resposta = resposta.choices[0].message.content
            st.session_state.mensagens.append({"role": "ai", "content": conteudo_resposta})

        except Exception as e:
            st.error("Erro na chamada da API")
            st.code(str(e))

    # mostrar histórico do chat
    st.markdown("### Histórico do Chat")
    for msg in st.session_state.mensagens:
        if msg["role"] == "user":
            st.markdown(f"**Você:** {msg['content']}")
        else:
            st.markdown(f"**IA:** {msg['content']}")
