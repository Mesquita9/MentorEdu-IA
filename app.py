import streamlit as st
import pdfplumber
import os
import numpy as np
import faiss

from groq import Groq
from sentence_transformers import SentenceTransformer

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

model = SentenceTransformer("all-MiniLM-L6-v2")

st.title("IA que conversa com PDF")

uploaded_file = st.file_uploader("Envie um PDF", type="pdf")

if uploaded_file:

    texto = ""

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                texto += page_text

    # dividir texto em partes
    chunk_size = 500
    chunks = [texto[i:i+chunk_size] for i in range(0, len(texto), chunk_size)]

    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    pergunta = st.text_input("Faça uma pergunta sobre o PDF")

    if pergunta:

        pergunta_embedding = model.encode([pergunta])

        D, I = index.search(np.array(pergunta_embedding), k=3)

        contexto = ""
        for i in I[0]:
            contexto += chunks[i] + "\n"

        prompt = f"""
Use apenas o contexto abaixo para responder.

Contexto:
{contexto}

Pergunta:
{pergunta}
"""

        resposta = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=400
        )

        st.write(resposta.choices[0].message.content)
