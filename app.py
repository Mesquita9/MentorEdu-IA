import streamlit as st
from pypdf import PdfReader
import os, faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. ESTILO INSTITUCIONAL IF (MINIMALISTA)
st.set_page_config(page_title="MentorEdu | IF", page_icon="🎓", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; color: #212529; }
    .main-title { text-align: center; color: #32a041; font-weight: 800; font-size: 2.2rem; }
    [data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #eee; }
    .stButton>button { background-color: #32a041 !important; color: white !important; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# 2. FUNÇÕES DE SUPORTE (CARREGAMENTO ÚNICO)
@st.cache_resource
def carregar_motores():
    try:
        chave = os.getenv("GROQ_API_KEY")
        return Groq(api_key=chave), SentenceTransformer("all-MiniLM-L6-v2")
    except:
        return None, None

client, model = carregar_motores()

# 3. INTERFACE LATERAL
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    st.markdown("---")
    area = st.selectbox("Disciplina:", ["Geral", "Matemática", "Física", "Química"])
    arquivo = st.file_uploader("Subir material (PDF)", type="pdf")
    if st.button("Limpar Tudo"):
        st.session_state.clear()
        st.rerun()

# 4. MEMÓRIA E PROCESSAMENTO (CACHE DE ALTA PERFORMANCE)
if "historico" not in st.session_state:
    st.session_state.historico = []
if "banco" not in st.session_state:
    st.session_state.banco = None

if arquivo and st.session_state.banco is None:
    with st.spinner("Otimizando leitura..."):
        leitor = PdfReader(arquivo)
        textos, paginas = [], []
        for i, pagina in enumerate(leitor.pages):
            conteudo = pagina.extract_text()
            if conteudo:
                blocos = [conteudo[i:i+500] for i in range(0, len(conteudo), 500)]
                for b in blocos:
                    if len(b.strip()) > 50:
                        textos.append(b.strip())
                        paginas.append(i + 1)
        
        if textos:
            vetores = model.encode(textos, show_progress_bar=False)
            index = faiss.IndexFlatL2(vetores.shape[1])
            index.add(np.array(vetores))
            st.session_state.banco = {"index": index, "textos": textos, "pgs": paginas}

# 5. CHAT STREAMING (RESPOSTA INSTANTÂNEA)
st.markdown('<h1 class="main-title">MentorEdu</h1>', unsafe_allow_html=True)

for m in st.session_state.historico:
    with st.chat_message(m["role"]):
        st.write(m["content"])

if pergunta := st.chat_input("Dúvida sobre o material?"):
    st.session_state.historico.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.write(pergunta)

    with st.chat_message("assistant"):
        contexto = ""
        if st.session_state.banco:
            v_pergunta = model.encode([pergunta])
            dist, indices = st.session_state.banco["index"].search(np.array(v_pergunta), k=2)
            for idx in indices[0]:
                contexto += f"[Pág {st.session_state.banco['pgs'][idx]}] {st.session_state.banco['textos'][idx]}\n\n"

        prompt_sistema = f"Você é o MentorEdu, tutor do IF. Área: {area}. Responda de forma didática e profissional."
        
        try:
            fluxo = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": prompt_sistema},
                    {"role": "user", "content": f"Contexto:\n{contexto}\n\nPergunta: {pergunta}"}
                ],
                stream=True
            )
            resposta_completa = st.write_stream(fluxo)
            st.session_state.historico.append({"role": "assistant", "content": resposta_completa})
        except:
            st.error("Erro na conexão com a IA. Verifique a API Key nos Secrets.")
