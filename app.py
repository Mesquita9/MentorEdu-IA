import streamlit as st
import pdfplumber
import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.title("IA que conversa com PDF")

uploaded_file = st.file_uploader("Envie um PDF", type="pdf")

if uploaded_file is not None:

    texto = ""

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                texto += page_text

    pergunta = st.text_input("Faça uma pergunta sobre o PDF")

    if pergunta:

        prompt = f"""
Responda usando apenas o conteúdo do PDF abaixo.

PDF:
{texto[:6000]}

Pergunta:
{pergunta}
"""

        resposta = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )

        st.write(resposta.choices[0].message.content)
