import streamlit as st
from google import genai
import PyPDF2

st.set_page_config(page_title="MentorEdu - IA", page_icon="🎓")

# Conectando com a chave nova que você colocou nos Secrets
try:
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
except:
    st.error("Configure a GOOGLE_API_KEY nos Secrets do Streamlit.")
    st.stop()

st.title("🎓 MentorEdu")
uploaded_file = st.file_uploader("Suba seu PDF de estudo", type="pdf")

if uploaded_file:
    # Lendo apenas as primeiras páginas para não estourar o limite
    reader = PyPDF2.PdfReader(uploaded_file)
    texto = ""
    for i in range(min(10, len(reader.pages))): # Limitamos a 10 páginas por vez
        texto += reader.pages[i].extract_text()
    
    st.success("PDF carregado (Modo Otimizado)!")
    pergunta = st.chat_input("O que deseja saber sobre este conteúdo?")

    if pergunta:
        try:
            # Enviando um comando bem curto para a IA não travar
            prompt = f"Contexto: {texto[:1000]}... Pergunta: {pergunta}"
            response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
            st.write(response.text)
        except:
            st.error("⚠️ O Gemini está ocupado. Espere 15 segundos e tente de novo.")
