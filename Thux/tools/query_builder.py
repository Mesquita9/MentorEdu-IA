"""
Construtor de buscas do Thux.

Este arquivo recebe uma pergunta natural do usuário e tenta transformar
essa pergunta em termos úteis para buscar na biblioteca.

A ideia não é criar um sistema engessado de respostas prontas.
A ideia é ajudar o Thux a procurar melhor nos PDFs.

Exemplo:
Pergunta:
"T, não entendi o que é função."

Termos gerados:
- conceito de função
- definição de função
- exemplos iniciais
- função
"""


def normalize_text(text: str) -> str:
    """
    Normaliza o texto para facilitar análise simples.

    Por enquanto:
    - deixa tudo em minúsculo;
    - remove espaços extras.
    """

    return " ".join(text.lower().strip().split())


def detect_discipline(question: str) -> str | None:
    """
    Tenta identificar a disciplina principal da pergunta.

    Retorno:
    - "Matemática"
    - "Física"
    - None, se não tiver certeza.
    """

    normalized = normalize_text(question)

    math_terms = [
        "função",
        "equação",
        "conjunto",
        "domínio",
        "imagem",
        "gráfico",
        "parábola",
        "raiz",
        "logaritmo",
        "trigonometria",
        "seno",
        "cosseno",
        "derivada",
        "integral",
    ]

    physics_terms = [
        "velocidade",
        "aceleração",
        "força",
        "massa",
        "energia",
        "trabalho",
        "potência",
        "queda livre",
        "inércia",
        "newton",
        "movimento",
        "cinemática",
        "dinâmica",
    ]

    math_score = sum(1 for term in math_terms if term in normalized)
    physics_score = sum(1 for term in physics_terms if term in normalized)

    if math_score > physics_score and math_score > 0:
        return "Matemática"

    if physics_score > math_score and physics_score > 0:
        return "Física"

    return None


def detect_level(question: str) -> str | None:
    """
    Tenta identificar o nível desejado da explicação.

    Retorno:
    - "Elementar"
    - "Avançado"
    - None, se não tiver certeza.
    """

    normalized = normalize_text(question)

    elementary_signals = [
        "ensino médio",
        "básico",
        "base",
        "simples",
        "iniciante",
        "começando",
        "sem linguagem rebuscada",
        "não entendi",
        "me explica do zero",
    ]

    advanced_signals = [
        "formal",
        "demonstração",
        "rigoroso",
        "universitário",
        "faculdade",
        "avançado",
        "prova difícil",
    ]

    if any(signal in normalized for signal in elementary_signals):
        return "Elementar"

    if any(signal in normalized for signal in advanced_signals):
        return "Avançado"

    return None


def build_search_terms(question: str) -> list[str]:
    """
    Gera termos de busca a partir de uma pergunta natural.

    Esta função ainda é simples, mas já melhora muito em relação
    a buscar apenas uma palavra solta.
    """

    normalized = normalize_text(question)

    terms = []

    # Funções
    if "função composta" in normalized:
        terms.extend([
            "função composta",
            "composição de funções",
            "g(f(x))",
        ])

    elif "função inversa" in normalized:
        terms.extend([
            "função inversa",
            "determinação da função inversa",
            "funções inversas",
        ])

    elif "domínio" in normalized or "imagem" in normalized:
        terms.extend([
            "domínio e imagem",
            "domínio",
            "imagem",
            "função",
        ])

    elif "função" in normalized:
        terms.extend([
            "conceito de função",
            "definição de função",
            "exemplos iniciais",
            "função",
        ])

    # Conjuntos
    elif "conjunto" in normalized or "conjuntos" in normalized:
        terms.extend([
            "conjuntos",
            "noção de conjunto",
            "elemento",
            "pertinência",
        ])

    # Física: cinemática
    elif "velocidade" in normalized:
        terms.extend([
            "velocidade",
            "velocidade média",
            "movimento",
            "cinemática",
        ])

    elif "aceleração" in normalized:
        terms.extend([
            "aceleração",
            "movimento uniformemente variado",
            "cinemática",
        ])

    elif "inércia" in normalized:
        terms.extend([
            "inércia",
            "primeira lei de Newton",
            "força",
        ])

    # Caso ainda não tenha identificado nada específico,
    # usa a pergunta inteira como termo geral.
    if not terms:
        terms.append(question)

    # Remove duplicatas sem perder a ordem.
    unique_terms = []

    for term in terms:
        if term not in unique_terms:
            unique_terms.append(term)

    return unique_terms


def build_query_plan(question: str) -> dict:
    """
    Cria um plano de busca completo.

    Retorna:
    - pergunta original;
    - disciplina detectada;
    - nível detectado;
    - termos de busca sugeridos.
    """

    return {
        "question": question,
        "discipline": detect_discipline(question),
        "level": detect_level(question),
        "search_terms": build_search_terms(question),
    }


if __name__ == "__main__":
    """
    Teste manual.

    Para rodar:
    python3 tools/query_builder.py
    """

    test_questions = [
        "T, não entendi o que é função.",
        "Me explica domínio e imagem sem linguagem rebuscada.",
        "Pinguim, como funciona função composta?",
        "Me explica inércia como se fosse ensino médio.",
    ]

    for question in test_questions:
        plan = build_query_plan(question)

        print("\nPergunta:")
        print(question)

        print("\nPlano de busca:")
        print(plan)

        print("-" * 80)
