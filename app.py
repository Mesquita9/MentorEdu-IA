import streamlit as st
from google import genai
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time

# Configuração da Página
st.set_page_config(page_title="MentorEdu - IA", page_icon="🎓")

# Acessando a chave de forma segura (Secrets do Streamlit)
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    client = genai.Client(api_key=api_key)
except:
    st.error("Erro: Chave API não configurada nos Secrets.")
    st.stop()

st.title("🎓 MentorEdu - Seu Professor IA")
st.write("Envie um PDF e faça perguntas específicas sobre o conteúdo.")

uploaded_file = st.file_uploader("Envie um PDF", type="pdf")

if uploaded_file:
    # Lendo o PDF
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    texto_completo = ""
    for page in pdf_reader.pages:
        texto_completo += page.extract_text()
    
    # DIVISÃO EM PEDAÇOS PEQUENOS (Ajustado para evitar o limite)
    # Aqui mudamos para 300 caracteres para ficar bem leve
    tamanho = 300 
    chunks = [texto_completo[i:i+tamanho] for i in range(0, len(texto_completo), tamanho)]
    
    st.success("PDF carregado com sucesso!")

    pergunta = st.chat_input("Digite sua pergunta sobre o PDF")

    if pergunta:
        with st.chat_message("user"):
            st.write(pergunta)
        
        with st.chat_message("assistant"):
            try:
                # BUSCA INTELIGENTE (RAG)
                vectorizer = TfidfVectorizer()
                vectors = vectorizer.fit_transform(chunks + [pergunta])
                
                # Pegamos apenas o pedaço mais relevante (top_k = 1)
                similarities = cosine_similarity(vectors[-1], vectors[:-1])
                melhor_chunk = chunks[np.argmax(similarities)]
                
                # Criando o contexto para a IA
                prompt = f"Baseado neste trecho do PDF: '{melhor_chunk}', responda de forma curta: {pergunta}"
                
                # CHAMADA DA IA
                response = client.models.generate_content(
                    model="gemini-1.5-flash", 
                    contents=prompt
                )
                
                st.write(response.text)
                # Pausa obrigatória para não estourar o limite na próxima
                time.sleep(5) 

            except Exception as e:
                st.error("⚠️ Limite da API atingido. Por favor, aguarde 30 segundos e tente novamente.")
