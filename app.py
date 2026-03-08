import streamlit as st
import pdfplumber, os, faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. ESTILO BLINDADO (FORÇA TEXTO BRANCO E FUNDO ESCURO)
st.set_page_config(page_title="MentorEdu", layout="wide")

st.markdown("""
    <style>
    /* Fundo principal e texto */
    .stApp { background-color: #0e1117 !important; color: #ffffff !important; }
    
    /* Força cor de todos os textos, parágrafos e spans */
    p, span, label, .stMarkdown, [data-testid="stWidgetLabel"] p {
        color: #ffffff !important;
    }

    /* Resolve caixas de input brancas/invisíveis */
    div[data-baseweb="select"] > div, 
    div[data-testid="stFileUploader"] section {
        background-color: #1c2128 !important;
        border: 1px solid #30363d !important;
        color: #ffffff !important;
    }
    
    /* Balões de Chat */
    [data-testid="stChatMessage"] {
        background-color: #1c2128 !important;
        border: 1px solid #3b424b !important;
        color: #ffffff !important;
    }

    /* Título */
    .main-header { color: #00d4ff; text-align: center; font-size: 2.5rem; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 2. CARREGAMENTO DOS MOTORES
@st.cache_resource
def setup_ia():
    try:
        # Puxa a chave dos Secrets ou do ambiente
        api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        client = Groq(api_key=api_key)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return client, model
    except Exception:
        return None, None

client, model = setup_ia()

# 3. BARRA LATERAL
with st.sidebar:
    # Carrega logo se existir no repositório
    if os.path.exists("logo.png"):
        st.image("logo.png", width=120)
    
    st.markdown("### 🧪 MENU DO RICK")
    
    # Personalidades conforme sua última instrução
    modo = st.selectbox("PERSONALIDADE:", ["Rick Acadêmico", "Rick Sarcástico"])
    
    up = st.file_uploader("📂 SUBIR PDF (BASE)", type="pdf")
    
    if st.button("LIMPAR SISTEMA"):
        st.session_state.chat = []
        st.rerun()

# 4. PROCESSAMENTO DE DADOS (MEMÓRIA)
if "chat" not in st.session_state:
    st.session_state.chat = []

chunks, pgs = [], []

if up and model:
    try:
        with pdfplumber.open(up) as pdf:
            for i, p in enumerate(pdf.pages):
                txt = p.extract_text()
                if txt:
                    for lin in txt.split('\n'):
                        if len(lin.strip()) > 35:
                            chunks.append(lin.strip())
                            pgs.append(i + 1)
        if chunks:
            embs = model.encode(chunks)
            idx = faiss.IndexFlatL2(embs.shape[1])
            idx.add(np.array(embs))
    except Exception:
        st.error("Erro ao ler o PDF. Verifique se o arquivo não está protegido.")

# 5. ÁREA DE INTERAÇÃO
st.markdown('<p class="main-header">MentorEdu</p>', unsafe_allow_html=True)

# Mostra histórico
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# Entrada do usuário
if p := st.chat_input("Diz aí, Morty..."):
    st.session_state.chat.append({"role": "user", "content": p})
    with st.chat_message("user"):
        st.write(p)

    with st.chat_message("assistant"):
        ctx = ""
        # Busca no PDF se houver um
        if up and chunks and model:
            try:
                D, I = idx.search(np.array(model.encode([p])), k=2)
                for i in I[0]:
                    ctx += f"[Pág {pgs[i]}] {chunks[i]}\n\n"
            except:
                pass

        # Configuração das personas
        prompts = {
            "Rick Acadêmico": "Você é o Rick Reitor do IFCE. Fale de forma acadêmica, técnica e cite normas da ABNT.",
            "Rick Sarcástico": "Você é o Rick Sanchez. Use sarcasmo, chame o usuário de Morty e seja impaciente, mas responda."
        }

        try:
            prompt_final = f"Contexto do PDF:\n{ctx}\n\nPergunta do Morty: {p}" if ctx else p
            
            chamada = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": prompts[modo]},
                    {"role": "user", "content": prompt_final}
                ]
            )
            resp = chamada.choices[0].message.content
            st.write(resp)
            st.session_state.chat.append({"role": "assistant", "content": resp})
        except Exception:
            st.error("Erro na comunicação com a Groq. Verifique sua API Key!")
