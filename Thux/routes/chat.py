"""
Rotas de chat do Thux.

Recebe mensagens do frontend e envia para o cérebro do Thux.

Ideia:
- o usuário não precisa falar de um jeito específico;
- mensagens casuais ou vagas vão para o Thux normal;
- perguntas de estudo vão para a biblioteca;
- se a biblioteca falhar, usa o Thux normal como fallback.
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


def is_likely_study_question(message: str) -> bool:
    """
    Decide se a mensagem parece uma pergunta de estudo.

    Importante:
    - Isso não deve exigir que o usuário fale bonito.
    - A função só tenta evitar usar biblioteca em conversa casual.
    """

    text = message.strip().lower()

    if not text:
        return False

    casual_exact_messages = {
        "oi",
        "olá",
        "ola",
        "opa",
        "eai",
        "e aí",
        "fala",
        "salve",
        "coe",
        "bom dia",
        "boa tarde",
        "boa noite",
        "tudo bom",
        "tudo bom?",
        "td bom",
        "td bom?",
        "blz",
        "beleza",
        "como vai",
        "como vai?",
        "testando",
        "teste",
    }

    if text in casual_exact_messages:
        return False

    casual_fragments = [
        "tudo bem",
        "tudo bom",
        "como você está",
        "como vc ta",
        "como vc tá",
        "bom dia",
        "boa tarde",
        "boa noite",
    ]

    for fragment in casual_fragments:
        if fragment in text and len(text) < 80:
            return False

    study_keywords = [
        "explique",
        "explica",
        "me explica",
        "não entendi",
        "nao entendi",
        "o que é",
        "o que e",
        "como funciona",
        "como resolver",
        "resolva",
        "calcule",
        "calcular",
        "defina",
        "definição",
        "definicao",
        "exemplo",
        "exemplos",
        "exercício",
        "exercicio",
        "questão",
        "questao",
        "prova",
        "atividade",
        "matemática",
        "matematica",
        "física",
        "fisica",
        "química",
        "quimica",
        "função",
        "funcao",
        "domínio",
        "dominio",
        "imagem",
        "conjunto",
        "força",
        "forca",
        "energia",
        "velocidade",
        "aceleração",
        "aceleracao",
        "inércia",
        "inercia",
        "newton",
        "ligação",
        "ligacao",
        "covalente",
        "iônica",
        "ionica",
        "sigma",
        "pi",
    ]

    for keyword in study_keywords:
        if keyword in text:
            return True

    # Perguntas longas normalmente têm intenção real, mesmo com erro de digitação.
    if len(text) >= 80 and "?" in text:
        return True

    # Frases muito curtas e vagas não devem chamar biblioteca.
    if len(text) <= 30:
        return False

    return False


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

    if not is_likely_study_question(user_message):
        try:
            response = ask_thux(user_message)

        except Exception as basic_error:
            print("\nAviso: falha no chat básico do Thux.")
            print(f"Erro: {basic_error}\n")

            response = "Fala, mestre. Manda o B.O. de hoje."

        return {
            "response": response
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
