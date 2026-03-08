import streamlit as st
import pdfplumber
import os
import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. CONFIGURAÇÃO VISUAL (ANTI-TEXTO ILEGÍVEL)
# ==========================================
st.set_page_config(page_title="MentorEdu", page_icon="🧪", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    
    /* Força fonte global e cor de fundo */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif !important;
        background-color: #0e1117 !important;
        color: #ffffff !important;
    }

    /* BARRA LATERAL (SIDEBAR) */
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    /* Força todo o texto da sidebar a ser branco */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    /* CAIXAS DE SELEÇÃO E UPLOADER (Fundo escuro, texto claro) */
    div[data-baseweb="select"] > div, 
    div[data-testid="stFileUploader"] section {
        background-color: #1e2530 !important;
        border: 1px solid #3b424b !important;
        color: #ffffff !important;
    }
    div[data-baseweb="select"] span { color: #ffffff !important; }

    /* BALÕES DE CHAT (Fundo mais claro que a tela, texto branco brilhante) */
    [data-testid="stChatMessage"] {
        background-color: #1c2128 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px;
        padding: 10px;
    }
    /* Força o texto de quem fala (Morty/Rick) a ser branco */
    [data-testid="stChatMessage"] div, [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] span {
        color: #ffffff !important;
    }

    /* TÍTULO PRINCIPAL */
    .main-title { 
        text-align: center; 
        background: linear-gradient(90deg, #00d4ff, #88e23b);
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 3rem; margin-top: -30px; 
    }
    
    header, footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. CARREGAMENTO DA INTELIGÊNCIA
# ==========================================
@st.cache_resource
def load_engine():
    try:
        api_key = os.getenv("GROQ_API_KEY")
        c = Groq(api_key=api_key)
        m = SentenceTransformer("all-MiniLM-L6-v2")
        return c, m
    except Exception as e:
        return None, None

client, model = load_engine()

# ==========================================
# 3. INTERFACE LATERAL (CONTROLES)
# ==========================================
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=120)
    
    st.markdown("### 🧪 MENTOREDU")
    
    # Personalidades limitadas ao combinado
    modo = st.selectbox("ESTILO DO RICK:", ["Rick Acadêmico", "Rick Sarcástico"])
    up = st.file_uploader("📂 SUBIR BASE (PDF)", type="pdf")
    
    if st.button("LIMPAR HISTÓRICO"):
        st.session_state.mensagens = []
        st.rerun()

# ==========================================
# 4. PROCESSAMENTO DO PDF E MEMÓRIA
# ==========================================
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

chunks, pgs = [], []

if up and model:
    with st.spinner("Rick está lendo o documento..."):
        try:
            with pdfplumber.open(up) as pdf:
                for i, p in enumerate(pdf.pages):
                    txt = p.extract_text()
                    if txt:
                        for line in txt.split('\n'):
                            if len(line.strip()) > 40:
                                chunks.append(line.strip())
                                pgs.append(i + 1)
            
            if chunks:
                embs = model.encode(chunks)
                idx = faiss.IndexFlatL2(embs.shape[1])
                idx.add(np.array(embs))
        except Exception as e:
            st.error("Erro ao processar o PDF. Tente outro arquivo.")

# ==========================================
# 5. ÁREA DE CONVERSA PRINCIPAL
# ==========================================
st.markdown('<h1 class="main-title">MentorEdu</h1>', unsafe_allow_html=True)

# Renderiza histórico de conversas
for m in st.session_state.mensagens:
    with st.chat_message(m["role"]):
        st.markdown(f"**{m['role'].upper()}:** {m['content']}")

# Caixa de input para o usuário
if prompt := st.chat_input("Diz aí, Morty..."):
    st.session_state.mensagens.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(f"**MORTY:** {prompt}")

    with st.chat_message("assistant"):
        ctx = ""
        # Busca contexto no PDF se houver um arquivo carregado
        if up and chunks and model:
            try:
                q_emb = model.encode([prompt])
                D, I = idx.search(np.array(q_emb), k=2)
                for i in I[0]:
                    ctx += f"[Pág {pgs[i]}] {chunks[i]}\n\n"
            except Exception:
                pass # Ignora erro silenciosamente se a busca falhar

        # Personalidades do sistema
        p_sys = {
            "Rick Acadêmico": "Você é o Rick Reitor do IFCE. Focado em normas da ABNT e ciência pura.",
            "Rick Sarcástico": "Você é o Rick Sanchez clássico. Sarcástico, genial e irritadiço."
        }

        # Conecta com a IA (Groq)
        try:
            full_p = f"Contexto do PDF:\n{ctx}\n\nPergunta: {prompt}" if ctx else prompt
            
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": p_sys[modo]},
                    {"role": "user", "content": full_p}
                ]
            )
            ans = res.choices[0].message.content
            
            st.markdown(f"**RICK:** {ans}")
            st.session_state.mensagens.append({"role": "assistant", "content": ans})
            
        except Exception as e:
            st.error("A conexão com o portal falhou! Verifique a chave GROQ_API_KEY no Streamlit Cloud.")
