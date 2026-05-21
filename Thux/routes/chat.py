"""
Rotas de chat do Thux.

Este arquivo recebe mensagens do frontend e envia para o cérebro do Thux.

Fluxo:
- usuário envia mensagem;
- Thux tenta responder usando a biblioteca própria;
- se a biblioteca falhar, usa o chat básico como fallback;
- retorna resposta em JSON.
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel

from core.brain import ask_thux, ask_thux_with_knowledge


router = APIRouter()


class ChatRequest(BaseModel):
    """
    Modelo esperado para mensagens do chat.

    Aceita:
    {
        "message": "texto do usuário"
    }
    """

    message: str | None = None
    text: str | None = None
    prompt: str | None = None


def extract_user_message(data: dict) -> str:
    """
    Extrai a mensagem do usuário aceitando nomes diferentes de campo.
    """

    user_message = (
        data.get("message")
        or data.get("text")
        or data.get("prompt")
    )

    if user_message is None:
        return ""

    return str(user_message).strip()


@router.post("/chat")
async def chat(request: Request):
    """
    Endpoint principal do chat.

    Espera receber JSON como:
    {
        "message": "texto do usuário"
    }

    Retorna:
    {
        "response": "resposta do Thux"
    }
    """

    try:
        data = await request.json()

    except Exception:
        return {
            "error": "JSON inválido.",
            "response": "Não consegui ler tua mensagem, pai. Manda em JSON certinho."
        }

    user_message = extract_user_message(data)

    if not user_message:
        return {
            "error": "Mensagem vazia.",
            "response": "A mensagem veio vazia, pai."
        }

    try:
        response = ask_thux_with_knowledge(user_message)

    except Exception as knowledge_error:
        print("\nAviso: falha ao usar biblioteca do Thux.")
        print(f"Erro: {knowledge_error}")
        print("Usando resposta básica como fallback.\n")

        response = ask_thux(user_message)

    return {
        "response": response
    }


@router.post("/api/chat")
async def api_chat(request: Request):
    """
    Alias para compatibilidade com frontends que chamem /api/chat.
    """

    return await chat(request)
