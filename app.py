import streamlit as st
import pdfplumber
import os
from groq import Groq

# pega a chave da variável de ambiente
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.title("IA que conversa com PDF")

uploaded_file = st.file_uploader("Envie um PDF", type="pdf")

if uploaded_file is not None:

    texto = ""

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            texto += page.extract_text()

    pergunta = st.text_input("Faça uma pergunta sobre o PDF")

    if pergunta:

        prompt = f"""
        Responda usando apenas o conteúdo do PDF abaixo.

        PDF:
        {texto[:12000]}

        Pergunta:
        {pergunta}
        """

        resposta = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192",
        )

        st.write(resposta.choices[0].message.content)
