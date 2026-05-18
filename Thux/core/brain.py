"""
Cérebro de comunicação do Thux.

Este arquivo é responsável por enviar mensagens para a API de IA
e devolver a resposta para o restante do sistema.

Importante:
- A personalidade do Thux vem de core/personality.py
- As configurações vêm de core/config.py
- A chave da API NÃO fica neste arquivo
"""

import os
import requests

from core.config import (
    AI_MODEL,
    GROQ_API_KEY_ENV,
    MAX_RESPONSE_TOKENS,
    TEMPERATURE,
)

from core.personality import SYSTEM_PROMPT


# URL oficial da API da Groq no formato compatível com OpenAI
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def get_groq_api_key():
    """
    Busca a chave da Groq nas variáveis de ambiente.

    Isso evita colocar chave secreta diretamente no código.
    """
    api_key = os.getenv(GROQ_API_KEY_ENV)

    if not api_key:
        raise ValueError(
            "Chave da Groq não encontrada. "
            f"Configure a variável de ambiente {GROQ_API_KEY_ENV}."
        )

    return api_key


def ask_thux(user_message: str) -> str:
    """
    Envia uma mensagem para o modelo de IA e retorna a resposta do Thux.

    Parâmetro:
    - user_message: mensagem enviada pelo usuário

    Retorno:
    - resposta em texto gerada pela IA
    """

    # Busca a chave da API com segurança
    api_key = get_groq_api_key()

    # Cabeçalhos necessários para autenticação e envio em JSON
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Corpo da requisição enviada para a Groq
    payload = {
        "model": AI_MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_RESPONSE_TOKENS,
    }

    # Envia a requisição para a API
    response = requests.post(
        GROQ_API_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )

    # Se a API responder com erro, isso ajuda a identificar o problema
    response.raise_for_status()

    # Converte a resposta da API para dicionário Python
    data = response.json()

    # Extrai apenas o texto final gerado pelo modelo
    return data["choices"][0]["message"]["content"]
