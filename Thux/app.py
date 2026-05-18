"""
Arquivo principal do Thux-AI.

Este arquivo inicia a aplicação FastAPI e conecta as rotas principais do projeto.
Ele deve continuar simples: a lógica pesada fica distribuída nas pastas core, routes, tools e database.
"""

from fastapi import FastAPI

from routes.chat import router as chat_router


# Criação da aplicação principal
app = FastAPI(
    title="Thux-AI",
    description="Assistente pessoal didático focado em Física, Matemática e apoio docente.",
    version="0.1.0"
)


# Conecta as rotas de chat ao aplicativo principal
app.include_router(chat_router)


@app.get("/")
def home():
    """
    Rota inicial do sistema.

    Serve para verificar se o Thux está online.
    """
    return {
        "status": "Thux online",
        "message": "O núcleo inicial do Thux-AI está funcionando."
    }


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
