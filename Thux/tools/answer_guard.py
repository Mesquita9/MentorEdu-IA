"""
Guardião de respostas do Thux.

Este arquivo faz uma revisão local simples, sem usar API.

Objetivo:
- detectar possíveis erros conceituais comuns;
- aplicar avisos/correções simples;
- reduzir dependência da revisão por IA;
- evitar que o Thux entregue respostas perigosas em Matemática/Física.

Importante:
Este arquivo não substitui um professor nem uma revisão profunda.
Ele é um filtro inicial anti-besteira.
"""


def detect_function_concept_issues(answer: str) -> list[str]:
    """
    Detecta possíveis problemas conceituais em respostas sobre funções.
    """

    lower_answer = answer.lower()

    issues = []

    dangerous_patterns = [
        "chamados de domínio e imagem",
        "a é o domínio e b é a imagem",
        "b é a imagem",
        "segundo conjunto chamado de imagem",
        "todo elemento de b precisa",
        "todo elemento do contradomínio precisa",
        "se você tiver a saída",
        "se tiver a saída",
        "encontra a entrada",
        "cada saída tem uma entrada",
        "domínio e imagem são os dois conjuntos",
    ]

    for pattern in dangerous_patterns:
        if pattern in lower_answer:
            issues.append(
                f"Possível confusão conceitual detectada: '{pattern}'"
            )

    return issues


def build_function_warning() -> str:
    """
    Cria uma correção curta para anexar quando houver risco conceitual.
    """

    return (
        "\n\n⚠️ Observação conceitual importante:\n"
        "Em uma função de A em B, A é o domínio e B é o contradomínio. "
        "A imagem não é necessariamente igual a B; a imagem é apenas a parte de B "
        "que realmente recebe valores da função. Além disso, uma função garante "
        "entrada → uma única saída, mas não garante necessariamente saída → uma única entrada. "
        "Essa última ideia aparece em casos especiais, como funções injetoras ou bijetoras."
    )


def ensure_source_mentioned(answer: str) -> str:
    """
    Garante que a resposta tenha alguma menção de fonte quando parecer usar biblioteca.
    """

    lower_answer = answer.lower()

    has_source = (
        "fonte:" in lower_answer
        or "fontes:" in lower_answer
        or "fundamentos_matematica_elementar_001.pdf" in lower_answer
        or "página" in lower_answer
    )

    if has_source:
        return answer

    return (
        answer
        + "\n\nFonte usada: biblioteca do Thux. "
        "Verifique o arquivo e a página retornados pela busca para referência precisa."
    )


def remove_excessive_repetition(answer: str) -> str:
    """
    Remove algumas repetições simples muito comuns.

    Esta função é propositalmente conservadora.
    Ela não tenta reescrever o texto inteiro.
    """

    repeated_phrases = [
        "Agora, vamos responder à sua pergunta!",
        "Você entende melhor o que é função agora?",
    ]

    cleaned_answer = answer

    for phrase in repeated_phrases:
        first_position = cleaned_answer.find(phrase)

        if first_position == -1:
            continue

        before = cleaned_answer[: first_position + len(phrase)]
        after = cleaned_answer[first_position + len(phrase):]

        after = after.replace(phrase, "")

        cleaned_answer = before + after

    return cleaned_answer


def apply_answer_guard(answer: str, topic_hint: str | None = None) -> str:
    """
    Aplica filtros locais na resposta do Thux.

    Parâmetros:
    - answer: resposta gerada pela IA;
    - topic_hint: dica opcional de assunto, exemplo: "função".

    Retorno:
    - resposta revisada localmente.
    """

    guarded_answer = answer.strip()

    guarded_answer = remove_excessive_repetition(guarded_answer)
    guarded_answer = ensure_source_mentioned(guarded_answer)

    should_check_function = False

    if topic_hint:
        should_check_function = "função" in topic_hint.lower()

    if "função" in guarded_answer.lower():
        should_check_function = True

    if should_check_function:
        issues = detect_function_concept_issues(guarded_answer)

        if issues:
            guarded_answer += build_function_warning()

    return guarded_answer


if __name__ == "__main__":
    """
    Teste manual.

    Para rodar:
    python3 tools/answer_guard.py
    """

    test_answer = """
Uma função é uma relação entre dois conjuntos chamados de domínio e imagem.
Se você tiver a saída, encontra a entrada correspondente.

Fonte: fundamentos_matematica_elementar_001.pdf, página 87.
"""

    print(apply_answer_guard(test_answer, topic_hint="função"))
