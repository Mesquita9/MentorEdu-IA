import streamlit as st
import pdfplumber
import os
import numpy as np
import faiss
import re

from groq import Groq
from sentence_transformers import SentenceTransformer

st.title("IA que conversa e consulta PDFs")

# Verifica API Key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY não encontrada nas Secrets.")
    st.stop()

client = Groq(api_key=api_key)

# Modelo de embeddings
modelo_embeddings = SentenceTransformer("all-MiniLM-L6-v2")

# Upload do PDF
uploaded_file = st.file_uploader("Envie um PDF (opcional)", type="pdf")

chunks = []
paginas = []

if uploaded_file:
    with pdfplumber.open(uploaded_file) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                partes = [page_text[j:j+500] for j in range(0, len(page_text), 500)]
                partes = [
                    p for p in partes
                    if len(p.strip()) > 50 and
                       not re.search(r"https?://", p, re.IGNORECASE) and
                       not re.search(r"Exercícios?|Questão|Capítulo", p, re.IGNORECASE)
                ]
                for p in partes:
                    chunks.append(p)
                    paginas.append(i + 1)

    if len(chunks) > 0:
        st.write(f"Texto extraído do PDF: {len(chunks)} trechos válidos.")

        # Criar embeddings
        embeddings = modelo_embeddings.encode(chunks)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))

# Iniciar histórico de chat
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

pergunta = st.text_input("Faça uma pergunta sobre o PDF ou qualquer assunto")

if pergunta:
    # Se PDF existe, tenta usar contexto
    if len(chunks) > 0:
        pergunta_embedding = modelo_embeddings.encode([pergunta])
        D, I = index.search(np.array(pergunta_embedding), k=5)

        contexto = ""
        for i in I[0]:
            contexto += f"(Página {paginas[i]}) {chunks[i]}\n"

        contexto = contexto[:3000]  # limitar tamanho

        prompt = f"""
Você é uma IA que responde com base no PDF abaixo,
mas se a pergunta não estiver no PDF, você pode responder normalmente.
Não invente informações sobre o PDF.

Contexto do PDF:
{contexto}

Pergunta:
{pergunta}
"""
    else:
        # Sem PDF, IA responde normalmente
        prompt = pergunta

    try:
        resposta = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Você é uma IA conversacional que pode usar PDFs como contexto."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400
        )

        conteudo_resposta = resposta.choices[0].message.content

        # Salvar pergunta e resposta no histórico
        st.session_state.mensagens.append({"role": "user", "content": pergunta})
        st.session_state.mensagens.append({"role": "assistant", "content": conteudo_resposta})

    except Exception as e:
        st.error("Erro na chamada da API")
        st.code(str(e))

# Mostrar histórico do chat
st.markdown("### Histórico do Chat")
for msg in st.session_state.mensagens:
    if msg["role"] == "user":
        st.markdown(f"**Você:** {msg['content']}")
    else:
        st.markdown(f"**IA:** {msg['content']}")
