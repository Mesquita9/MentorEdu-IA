import streamlit as st
import pdfplumber, os, faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. DESIGN INSTITUCIONAL (IF)
st.set_page_config(page_title="MentorEdu | Instituto Federal", page_icon="🎓", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* Cores Institucionais */
    :root {
        --if-verde: #32a041;
        --if-vermelho: #e40613;
        --fundo: #fcfcfc;
    }

    .stApp { 
        background-color: var(--fundo); 
        color: #212529; 
        font-family: 'Inter', sans-serif; 
    }

    /* BARRA LATERAL */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e0e0e0;
    }
    
    /* ESTILIZAÇÃO DE BOTÕES */
    .stButton>button {
        background-color: var(--if-verde) !important;
        color: white !important;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton>button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }

    /* BALÕES DE CHAT ESTILO DIDÁTICO */
    [data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 1px solid #efefef !important;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        margin-bottom: 15px;
    }

    /* TÍTULOS */
    .main-title { 
        text-align: center; 
        color: var(--if-verde);
        font-weight: 800; font-size: 2.8rem; margin-top: -30px; 
    }
    
    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 40px;
    }

    header, footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

# 2. CONFIGURAÇÃO DO MOTOR DE IA
@st.cache_resource
def load_engine():
    try:
        api_key = os.getenv("GROQ_API_KEY")
        c = Groq(api_key=api_key)
        m = SentenceTransformer("all-MiniLM-L6-v2")
        return c, m
    except: return None, None

client, model = load_engine()

# 3. PAINEL LATERAL
with st.sidebar:
    # Busca a imagem logo.png que você mencionou
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    else:
        st.markdown("<h2 style='color: #32a041; text-align: center;'>IF</h2>", unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("### 📘 Configurações de Estudo")
    
    disciplina = st.selectbox("ÁREA DE CONHECIMENTO:", 
                             ["Geral", "Matemática", "Física", "Química", "Biologia", "Português"])
    
    uploaded_file = st.file_uploader("📂 CARREGAR MATERIAL (PDF)", type="pdf")
    
    if st.button("Limpar Conversa"):
        st.session_state.chat_history = []
        st.rerun()

# 4. PROCESSAMENTO DE DOCUMENTOS
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

chunks, pgs = [], []
if uploaded_file and model:
    with st.spinner("Analisando material didático..."):
        with pdfplumber.open(uploaded_file) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    for line in text.split('\n'):
                        if len(line.strip()) > 50:
                            chunks.append(line.strip()); pgs.append(i+1)
        if chunks:
            embeddings = model.encode(chunks)
            index_faiss = faiss.IndexFlatL2(embeddings.shape[1])
            index_faiss.add(np.array(embeddings))

# 5. INTERFACE PRINCIPAL
st.markdown('<h1 class="main-title">MentorEdu</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Assistente de Aprendizagem do Instituto Federal</p>', unsafe_allow_html=True)

# Renderização do Histórico
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrada do Usuário
if user_input := st.chat_input("Em que posso ajudar nos seus estudos hoje?"):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        context_text = ""
        if uploaded_file and chunks:
            query_emb = model.encode([user_input])
            D, I = index_faiss.search(np.array(query_emb), k=3)
            for idx in I[0]:
                context_text += f"[Ref. Pág {pgs[idx]}] {chunks[idx]}\n\n"

        # Prompt do Sistema: Foco Profissional e Didático
        sys_prompt = f"""Você é o MentorEdu, o assistente virtual oficial do Instituto Federal.
        Sua especialidade hoje é: {disciplina}.
        
        Instruções de Resposta:
        1. Seja profissional, cordial e didático.
        2. Utilize termos técnicos corretamente, mas explique-os se forem complexos.
        3. Se houver contexto de um PDF, use-o para fundamentar a resposta citando as páginas.
        4. Mantenha um tom encorajador para o aprendizado do aluno.
        5. Nunca utilize gírias ou sarcasmo."""

        try:
            prompt_payload = f"Contexto do Material:\n{context_text}\n\nDúvida do Aluno: {user_input}" if context_text else user_input
            
            chat_completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt_payload}
                ]
            )
            
            final_response = chat_completion.choices[0].message.content
            st.markdown(final_response)
            st.session_state.chat_history.append({"role": "assistant", "content": final_response})
            
        except Exception:
            st.error("Ocorreu um erro na conexão com o servidor de IA. Verifique sua chave API.")
