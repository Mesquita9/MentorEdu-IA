import streamlit as st
from google import genai
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time

# API
client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

st.set_page_config(page_title="MentorEdu", page_icon="👨‍🏫")

st.title("👨‍🏫 MentorEdu")
st.write("Envie um PDF e faça perguntas sobre o conteúdo.")

# memória
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

uploaded_file = st.file_uploader("Envie um PDF", type="pdf")

# -------------------------
# PROCESSAR PDF
# -------------------------

if uploaded_file:

    reader = PdfReader(uploaded_file)

    texto = ""
    for page in reader.pages:
        texto += page.extract_text()

    # dividir texto em pedaços
    tamanho = 300
    chunks = [texto[i:i+tamanho] for i in range(0, len(texto), tamanho)]

    st.session_state.chunks = chunks

    st.success("PDF carregado!")

# -------------------------
# MOSTRAR CHAT
# -------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# PERGUNTA
# -------------------------

if pergunta := st.chat_input("Digite sua pergunta"):

    st.session_state.messages.append(
        {"role": "user", "content": pergunta}
    )

    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):

        with st.spinner("Pensando..."):

            try:

                chunks = st.session_state.chunks

                contexto = ""

                if chunks:

                    vectorizer = TfidfVectorizer()

                    vetores = vectorizer.fit_transform(
                        chunks + [pergunta]
                    )

                    similaridade = cosine_similarity(
                        vetores[-1], vetores[:-1]
                    )

                    indice = np.argmax(similaridade)

                    contexto = chunks[indice]

                prompt = f"""
Você é um professor didático.

Use o contexto abaixo para responder.

Contexto do PDF:
{contexto}

Pergunta:
{pergunta}

Explique de forma simples e dê exemplos.
"""

                time.sleep(4)

                response = client.models.generate_content(
                    model="models/gemini-2.0-flash",
                    contents=prompt
                )

                resposta = response.text

                st.markdown(resposta)

            except Exception:

                resposta = "⚠️ Limite da API atingido. Aguarde alguns segundos."

                st.error(resposta)

    st.session_state.messages.append(
        {"role": "assistant", "content": resposta}
    )
