import streamlit as st
from google import genai
import PyPDF2

st.set_page_config(page_title="MentorEdu - IA", page_icon="🎓")

try:
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception as e:
    st.error(f"Erro ao carregar a chave: {e}")
    st.stop()

st.title("🎓 MentorEdu (Modo Diagnóstico)")
uploaded_file = st.file_uploader("Suba seu PDF", type="pdf")

if uploaded_file:
    reader = PyPDF2.PdfReader(uploaded_file)
    texto = ""
    # Lendo só as 3 primeiras páginas para ser ultra rápido
    for i in range(min(3, len(reader.pages))): 
        texto += reader.pages[i].extract_text()
    
    st.success("PDF carregado!")
    pergunta = st.chat_input("Digite algo")

    if pergunta:
        try:
            prompt = f"Contexto: {texto[:500]}... Pergunta: {pergunta}"
            response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
            st.write(response.text)
        except Exception as e:
            # O PULO DO GATO: MOSTRAR O ERRO REAL EM VEZ DA MENSAGEM GENÉRICA
            st.error(f"🚨 ERRO REAL DETECTADO: {e}")
