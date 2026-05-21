"""
Cérebro de comunicação do Thux.

Este arquivo é responsável por:
- conversar com a API da Groq;
- aplicar a personalidade do Thux;
- consultar a biblioteca de PDFs;
- montar uma resposta didática com base nos trechos encontrados;
- passar a resposta por uma revisão local simples antes de entregar ao usuário.

Importante:
- A personalidade vem de core/personality.py
- As configurações vêm de core/config.py
- A chave da API NÃO fica neste arquivo
- A biblioteca vem de tools/knowledge_search.py
- A revisão local vem de tools/answer_guard.py
"""

import os
import sys
import requests

from dotenv import load_dotenv


# Carrega variáveis do arquivo .env local.
# override=True garante que o .env local tenha prioridade no teste local.
load_dotenv(override=True)


# Caminhos principais do projeto.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(BASE_DIR, "tools")


# Garante que a raiz do projeto seja reconhecida.
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)


# Garante que a pasta tools possa ser importada.
if TOOLS_DIR not in sys.path:
    sys.path.append(TOOLS_DIR)


from core.config import (
    AI_MODEL,
    GROQ_API_KEY_ENV,
    MAX_RESPONSE_TOKENS,
    TEMPERATURE,
)

from core.personality import SYSTEM_PROMPT

from knowledge_search import search_knowledge_base
from answer_guard import apply_answer_guard


# URL da API da Groq no padrão compatível com OpenAI.
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


def call_groq(messages: list[dict]) -> str:
    """
    Envia mensagens para a API da Groq e retorna a resposta em texto.

    Parâmetro:
    - messages: lista de mensagens no formato system/user/assistant.

    Retorno:
    - texto gerado pelo modelo.
    """

    api_key = get_groq_api_key()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": AI_MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_RESPONSE_TOKENS,
    }

    response = requests.post(
        GROQ_API_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    return data["choices"][0]["message"]["content"]


def ask_thux(user_message: str) -> str:
    """
    Envia uma mensagem simples para o Thux, sem consultar a biblioteca.

    Esta função mantém o chat básico funcionando.
    """

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

    return call_groq(messages)


def format_knowledge_context(search_output: dict, max_results: int = 3) -> str:
    """
    Transforma os resultados da biblioteca em um contexto textual
    que será enviado para a IA.

    O objetivo é dar ao Thux trechos confiáveis para responder
    com base nos materiais do Google Drive.
    """

    results = search_output.get("results", [])

    if not results:
        return "Nenhum trecho relevante foi encontrado na biblioteca."

    context_parts = []

    for index, result in enumerate(results[:max_results], start=1):
        context_parts.append(
            f"""
Fonte {index}:
Disciplina: {result["discipline"]}
Nível: {result["level"]}
Arquivo: {result["file_name"]}
Página: {result["page"]}
Termo encontrado: {result["term"]}
Relevância: {result["weighted_score"]}

Trecho:
{result["excerpt"]}
"""
        )

    return "\n".join(context_parts)


def build_knowledge_prompt(query_plan: dict, knowledge_context: str) -> str:
    """
    Cria o prompt principal para resposta com biblioteca.

    Este prompt orienta o Thux a responder com didática e rigor.
    A revisão local extra é feita depois pelo answer_guard.py.
    """

    return f"""
Você responderá à pergunta do usuário usando, quando possível, os trechos da biblioteca do Thux.

Identidade de resposta:
- Você é o Thux: direto, didático, humano, crítico e informal quando couber.
- Pode usar humor leve, ironia moderada e palavrões ocasionais para quebrar tensão ou comemorar avanços.
- Mas em explicações de Matemática e Física, o rigor vem antes da personalidade.
- A piada nunca deve esconder incerteza, erro ou falta de fonte.
- Evite soar corporativo, artificial ou excessivamente fofo.
- Não chame o usuário de "Pinguim". Pinguim é um apelido do próprio assistente Thux, não do usuário.
- Não fique repetindo apelidos do usuário no começo e no fim da resposta.

Regras obrigatórias de uso da biblioteca:
1. Use os trechos da biblioteca como base principal.
2. Cite as fontes usadas com nome do arquivo e página.
3. A citação da fonte é obrigatória quando a biblioteca for usada.
4. Não invente páginas, autores, livros ou referências.
5. Não copie longos trechos do livro; explique com suas próprias palavras.
6. Se os trechos não forem suficientes, diga isso claramente.
7. Se houver ambiguidade, explique o limite da resposta.
8. Se o livro usar uma propriedade sem explicar, explique a entrelinha.
9. Não altere a definição matemática/física do material.
10. Não misture Física e Matemática sem necessidade.
11. Não transforme caso particular em regra geral.

Regras de rigor conceitual:
1. Preserve a definição original do conceito.
2. Diferencie definição, exemplo, propriedade e caso especial.
3. Não diga que uma propriedade especial é obrigatória para todos os casos.
4. Se uma condição vale apenas em um tipo específico, diga claramente que é um caso especial.
5. Quando houver termos parecidos, explique a diferença entre eles.
6. Não invente exigências além das que aparecem na definição.
7. Se perceber que uma explicação comum pode gerar confusão, avise e corrija.
8. Se os trechos mostrarem uma definição formal, traduza para linguagem simples sem mudar o sentido.
9. Não use exemplos que contradigam ou enfraqueçam a definição.
10. Prefira exemplos estáveis, objetivos e fáceis de verificar.

Travas importantes para Matemática:
1. Em uma função de A em B, cada elemento de A deve estar associado a exatamente um elemento de B.
2. Não diga que todo elemento de B precisa ser atingido para ser função.
3. Todo elemento de B ser atingido é propriedade de função sobrejetora, não de função em geral.
4. Diferencie sempre que necessário:
   - domínio: conjunto de partida;
   - contradomínio: conjunto de chegada;
   - imagem: parte do contradomínio que realmente é atingida pela função.
5. Não diga que imagem é igual ao contradomínio como regra geral.
6. Se em um exemplo todos os elementos do contradomínio forem atingidos, diga: "nesse exemplo, a imagem coincide com o contradomínio".
7. Se falar de função injetora, sobrejetora ou bijetora, diga que são tipos especiais de função.
8. Se explicar domínio e imagem, deixe claro que a imagem depende dos valores que a função realmente assume.
9. Não diga que uma saída permite descobrir uma única entrada, a menos que esteja falando explicitamente de função injetora ou bijetora.
10. A função comum garante entrada -> uma saída. Ela não garante, em geral, saída -> uma única entrada.

Regras para exemplos:
1. Use exemplos concretos, estáveis e objetivos.
2. Para função, bons exemplos são:
   - número -> dobro;
   - aluno -> nota;
   - pessoa -> idade;
   - país -> capital;
   - produto -> preço.
3. Evite exemplos subjetivos ou ambíguos para definições matemáticas.
4. Não use exemplos onde uma mesma entrada possa ter várias respostas, a menos que esteja explicando por que isso NÃO é função.
5. Depois do exemplo, volte para a definição formal e mostre a ligação entre os dois.
6. Se usar conjuntos A e B, explique o que é A, o que é B, o que é domínio, o que é contradomínio e qual é a imagem.

Regras didáticas:
1. Comece acolhendo a dúvida quando fizer sentido.
2. Explique como professor, não como enciclopédia.
3. Use linguagem simples antes da linguagem formal.
4. Quando usar fórmula ou notação, explique o significado.
5. Mostre o ponto central do conceito.
6. Dê um exemplo simples.
7. Aponte erros comuns ou confusões prováveis.
8. Se a pergunta for básica, não puxe conteúdo avançado sem necessidade.
9. Explique as entrelinhas: propriedades usadas, passos omitidos e por que eles são válidos.
10. Se corrigir uma confusão, faça isso de forma clara, sem humilhar.

Formato obrigatório da resposta quando usar biblioteca:
- Primeiro: acolha a dúvida em uma frase curta.
- Depois: explique a ideia central.
- Depois: apresente a definição de forma simples.
- Depois: dê um exemplo objetivo.
- Depois: explique uma confusão comum, se existir.
- Depois: cite as fontes usadas com arquivo e página.
- Se fizer sentido, finalize com uma pergunta ou sugestão de próximo passo.

Plano de busca:
Pergunta original: {query_plan["question"]}
Disciplina detectada: {query_plan["discipline"]}
Nível detectado: {query_plan["level"]}
Termos usados: {query_plan["search_terms"]}

Trechos encontrados na biblioteca:
{knowledge_context}
"""


def ask_thux_with_knowledge(user_message: str) -> str:
    """
    Responde usando a biblioteca própria do Thux.

    Fluxo:
    - recebe a pergunta do usuário;
    - consulta a biblioteca no Google Drive;
    - pega os trechos mais relevantes;
    - gera uma resposta inicial com a Groq;
    - passa a resposta por uma revisão local simples no answer_guard.py;
    - entrega a resposta final.
    """

    search_output = search_knowledge_base(
        question=user_message,
        max_files=1,
        max_results_per_term=2,
    )

    query_plan = search_output["query_plan"]

    knowledge_context = format_knowledge_context(
        search_output=search_output,
        max_results=3,
    )

    knowledge_prompt = build_knowledge_prompt(
        query_plan=query_plan,
        knowledge_context=knowledge_context,
    )

    draft_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "system",
            "content": knowledge_prompt,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

    draft_answer = call_groq(draft_messages)

    guarded_answer = apply_answer_guard(
        answer=draft_answer,
        topic_hint=user_message,
    )

    return guarded_answer


if __name__ == "__main__":
    """
    Teste manual local.

    Para rodar:
    python3 core/brain.py

    Atenção:
    - precisa ter GROQ_API_KEY configurada no .env local;
    - precisa ter credentials/google_drive_credentials.json no projeto local;
    - precisa ter a biblioteca do Google Drive compartilhada com a conta de serviço.
    """

    test_question = "T, não entendi o que é função. Me explica como se fosse ensino médio."

    response = ask_thux_with_knowledge(test_question)

    print("\nResposta do Thux:\n")
    print(response)
