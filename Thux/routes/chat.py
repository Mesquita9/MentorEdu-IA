"""
Rota de chat do Thux.

Este arquivo recebe mensagens do usuário e devolve respostas geradas pelo núcleo do Thux.
Ele não define a personalidade e não conversa diretamente com a Groq.
Quem faz isso é o core/brain.py.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.brain import ask_thux


# Cria um roteador separado para as rotas de chat
router = APIRouter()


class ChatRequest(BaseModel):
    """
    Modelo da mensagem recebida pelo chat.

    Exemplo esperado:
    {
        "message": "Me explica função afim"
    }
    """

    message: str


class ChatResponse(BaseModel):
    """
    Modelo da resposta enviada pelo Thux.

    Exemplo de saída:
    {
        "response": "Função afim é..."
    }
    """

    response: str


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Recebe uma mensagem do usuário e retorna uma resposta do Thux.
    """

    try:
        # Envia a mensagem para o cérebro do Thux
        response = ask_thux(request.message)

        # Devolve a resposta no formato esperado
        return ChatResponse(response=response)

    except Exception as error:
        # Se algo der errado, devolve erro claro para facilitar correção
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao gerar resposta do Thux: {str(error)}"
        )
