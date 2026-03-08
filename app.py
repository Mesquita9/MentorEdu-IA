import streamlit as st
from google import genai
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(page_title="MentorEdu - IA", page_icon="🎓")

try:
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception as e:
    st.error(f"Erro na chave: {e}")
    st.stop()

st.title("🎓 MentorEdu - Professor IA")
uploaded_file = st.file_uploader("Suba seu PDF de estudo", type="pdf")

if uploaded_file:
    # Lendo o PDF inteiro de forma inteligente
    reader = PyPDF2.PdfReader(uploaded_file)
    texto_completo = ""
    for page in reader.pages:
        texto_completo += page.extract_text()
    
    # Dividindo em pedaços (Chunks)
    tamanho = 500 
    chunks = [texto_completo[i:i+tamanho] for i in range(0, len(texto_completo), tamanho)]
    
    st.success("PDF carregado com sucesso!")
    pergunta = st.chat_input("Digite sua pergunta sobre o PDF")

    if pergunta:
        with st.chat_message("user"):
            st.write(pergunta)
        
        with st.chat_message("assistant"):
            try:
                # Buscando a melhor parte do PDF para responder
                vectorizer = TfidfVectorizer()
                vectors = vectorizer.fit_transform(chunks + [pergunta])
                similarities = cosine_similarity(vectors[-1], vectors[:-1])
                melhor_chunk = chunks[np.argmax(similarities)]
                
                prompt = f"Baseado neste texto do PDF: '{melhor_chunk}', responda a pergunta: {pergunta}"
                
                # O SEGREDO REVELADO: Usando o gemini-2.0-flash
                response = client.models.generate_content(
                    model="gemini-2.0-flash", 
                    contents=prompt
                )
                st.write(response.text)
            except Exception as e:
                # Agora, se der erro, ele mostra a verdade!
                st.error(f"🚨 ERRO: {e}")
