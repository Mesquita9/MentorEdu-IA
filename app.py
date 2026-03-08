import streamlit as st
import pdfplumber
import os
import numpy as np
import faiss
import re
from groq import Groq
from sentence_transformers import SentenceTransformer

# -------------------------------
# Título do app
st.markdown("## 🤖 IA Acadêmica com PDFs")
st.markdown("---")

# -------------------------------
# Verifica API Key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY não encontrada nas Secrets.")
    st.stop()

client = Groq(api_key=api_key)

# -------------------------------
# Modelo de embeddings
modelo_embeddings = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# Determinar cores legíveis dependendo do tema
theme_base = st.get_option("theme.base")  # 'light' ou 'dark'
user_bg = "#cce5ff" if theme_base == "light" else "#005f87"
ia_bg = "#e8e8e8" if theme_base == "light" else "#333333"
text_color = "#000000" if theme_base == "light" else "#ffffff"

# -------------------------------
# Layout em colunas para minimalismo
col1, col2 = st.columns([1, 4])

chunks = []
paginas = []
topicos = []

with col1:
    # Upload de PDF opcional e escondido
    with st.expander("📄 Carregar PDF (opcional)"):
        uploaded_file = st.file_uploader("", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        with pdfplumber.open(uploaded_file) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    # Extrair títulos como possíveis tópicos
                    linhas = page_text.split("\n")
                    for linha in linhas:
                        linha_limpa = linha.strip()
                        if len(linha_limpa) < 5:
                            continue
                        if re.match(r"^[A-ZÁÉÍÓÚ\s]{3,}", linha_limpa) or linha_limpa.endswith(":"):
                            topico = linha_limpa
                        else:
                            topico = ""
                        # Dividir texto em trechos
                        partes = [linha_limpa[j:j+500] for j in range(0, len(linha_limpa), 500)]
                        for p in partes:
                            if len(p.strip()) > 50 and not re.search(r"https?://", p):
                                chunks.append(p)
                                paginas.append(i + 1)
                                topicos.append(topico)

        if len(chunks) > 0:
            st.markdown(f"<small>{len(chunks)} trechos extraídos do PDF</small>", unsafe_allow_html=True)
            # Criar embeddings
            embeddings = modelo_embeddings.encode(chunks)
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(embeddings))

# -------------------------------
with col2:
    # Inicializa histórico de chat
    if "mensagens" not in st.session_state:
        st.session_state.mensagens = []

    # Input minimalista
    pergunta = st.text_input("💬 Pergunta", placeholder="Digite sua pergunta aqui...")

    if pergunta:
        # Se PDF existe, busca contexto
        if len(chunks) > 0:
            pergunta_embedding = modelo_embeddings.encode([pergunta])
            D, I = index.search(np.array(pergunta_embedding), k=5)

            contexto = ""
            for i in I[0]:
                topico_texto = f" (Tópico: {topicos[i]})" if topicos[i] else ""
                contexto += f"(Página {paginas[i]}{topico_texto}) {chunks[i]}\n"

            contexto = contexto[:3000]  # limitar tamanho

            prompt = f"""
Você é uma IA acadêmica que responde apenas com base no PDF abaixo.
Se a pergunta não estiver no PDF, você pode responder normalmente.
Sempre cite o número da página e o tópico se possível.
Forneça respostas claras, concisas e acadêmicas.

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
                    {"role": "system", "content": "Você é uma IA acadêmica que pode usar PDFs como referência."},
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

    # -------------------------------
    # Mostrar histórico do chat estilizado e legível
    for msg in st.session_state.mensagens:
        if msg["role"] == "user":
            st.markdown(
                f"<div style='background-color:{user_bg};padding:8px;margin:4px 0;border-radius:5px;color:{text_color}'><b>Você:</b> {msg['content']}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='background-color:{ia_bg};padding:8px;margin:4px 0;border-radius:5px;color:{text_color}'><b>IA:</b> {msg['content']}</div>",
                unsafe_allow_html=True
            )
