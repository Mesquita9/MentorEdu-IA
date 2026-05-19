"""
Arquivo principal do Thux-AI.

Este arquivo inicia a aplicação FastAPI, conecta as rotas principais
e entrega a interface visual do projeto.

Ele deve continuar simples: a lógica pesada fica distribuída nas pastas
core, routes, tools e database.
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from routes.chat import router as chat_router


# Criação da aplicação principal
app = FastAPI(
    title="Thux-AI",
    description="Assistente pessoal didático focado em Física, Matemática e apoio docente.",
    version="0.1.0"
)


# Conecta as rotas de chat ao aplicativo principal
app.include_router(chat_router)


# Entrega os arquivos estáticos da interface
# Exemplo: /frontend/style.css e /frontend/script.js
app.mount(
    "/frontend",
    StaticFiles(directory="frontend"),
    name="frontend"
)


@app.get("/")
def home():
    """
    Rota inicial do sistema.

    Agora ela entrega a interface visual do Thux.
    """
    return FileResponse("frontend/index.html")


@app.get("/health")
def health_check():
    """
    Rota de verificação da saúde do sistema.

    Útil para testes locais e futuramente para deploy em nuvem.
    """
    return {
        "status": "ok",
        "service": "Thux-AI"
    }
