"""
Rotas de chat do Thux.

Recebe mensagens do frontend e envia para o cérebro do Thux.

Fluxo:
- se for conversa casual, responde localmente rápido;
- se for pergunta de estudo, tenta usar biblioteca;
- se a biblioteca falhar, usa chat básico como fallback.
"""

from fastapi import APIRouter, Request

from core.brain import ask_thux, ask_thux_with_knowledge


router = APIRouter()


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


def is_casual_message(message: str) -> bool:
    """
    Detecta mensagens simples que não precisam consultar biblioteca.
    """

    normalized = message.strip().lower()

    casual_messages = {
        "oi",
        "olá",
        "ola",
        "opa",
        "eai",
        "e aí",
        "fala",
        "fala ai",
        "fala aí",
        "bom dia",
        "boa tarde",
        "boa noite",
        "salve",
        "coe",
        "ei",
        "hey",
        "hello",
        "hi",
        "fala, krl",
        "fala krl",
    }

    if normalized in casual_messages:
        return True

    if len(normalized) <= 4 and not any(char.isdigit() for char in normalized):
        return True

    return False


def answer_casual_message(message: str) -> str:
    """
    Resposta local rápida para não gastar API nem consultar Drive.
    """

    return "Fala, mestre. Manda o B.O. de hoje."


@router.post("/chat")
async def chat(request: Request):
    """
    Endpoint principal do chat.
    """

    try:
        data = await request.json()

    except Exception:
        return {
            "error": "JSON inválido.",
            "response": "Não consegui ler tua mensagem. Manda de novo."
        }

    user_message = extract_user_message(data)

    if not user_message:
        return {
            "error": "Mensagem vazia.",
            "response": "A mensagem veio vazia, pai."
        }

    if is_casual_message(user_message):
        return {
            "response": answer_casual_message(user_message)
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
