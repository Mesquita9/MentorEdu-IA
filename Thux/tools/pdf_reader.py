"""
Leitor de PDFs do Thux.

Este arquivo é responsável por abrir arquivos PDF, extrair texto por página
e buscar termos dentro dos materiais.

Ele será usado futuramente para:
- ler livros e apostilas;
- identificar páginas relevantes;
- alimentar a base de conhecimento;
- permitir respostas com referência de página;
- encontrar teoria, exemplos, exercícios e resoluções.
"""

import fitz  # PyMuPDF


def open_pdf(pdf_path: str):
    """
    Abre um arquivo PDF.

    Parâmetro:
    - pdf_path: caminho do arquivo PDF no computador.

    Retorno:
    - documento PDF aberto.
    """

    return fitz.open(pdf_path)


def get_page_count(pdf_path: str) -> int:
    """
    Retorna a quantidade de páginas de um PDF.
    """

    document = open_pdf(pdf_path)
    page_count = len(document)
    document.close()

    return page_count


def extract_text_from_page(pdf_path: str, page_number: int) -> str:
    """
    Extrai o texto de uma página específica do PDF.

    Importante:
    - Para o usuário, a página começa em 1.
    - Para o Python/PyMuPDF, a página começa em 0.
    """

    document = open_pdf(pdf_path)

    page_index = page_number - 1

    if page_index < 0 or page_index >= len(document):
        document.close()
        raise ValueError("Número de página fora do intervalo do PDF.")

    page = document[page_index]
    text = page.get_text()

    document.close()

    return text


def extract_pdf_preview(pdf_path: str, page_number: int = 1, max_chars: int = 1200) -> str:
    """
    Extrai uma prévia do texto de uma página.

    Isso é útil para testar rapidamente se o PDF está sendo lido.
    """

    text = extract_text_from_page(pdf_path, page_number)

    return text[:max_chars]


def create_text_excerpt(text: str, search_term: str, context_chars: int = 450) -> str:
    """
    Cria um trecho ao redor do termo encontrado.

    Em vez de devolver a página inteira, devolve um pedaço do texto
    próximo ao termo buscado.
    """

    lower_text = text.lower()
    lower_term = search_term.lower()

    position = lower_text.find(lower_term)

    if position == -1:
        return ""

    start = max(position - context_chars, 0)
    end = min(position + len(search_term) + context_chars, len(text))

    excerpt = text[start:end].strip()

    return excerpt


def count_term_occurrences(text: str, search_term: str) -> int:
    """
    Conta quantas vezes o termo aparece no texto da página.
    """

    return text.lower().count(search_term.lower())


def looks_like_summary_page(text: str) -> bool:
    """
    Tenta identificar se uma página parece sumário/índice.

    Sumários costumam ter:
    - muitos pontinhos;
    - muitos números de página;
    - palavras como sumário, capítulo, índice;
    - várias linhas curtas com títulos e páginas.
    """

    lower_text = text.lower()

    dot_count = text.count("...")
    long_dot_count = text.count("......")

    summary_words = [
        "sumário",
        "índice",
        "capítulo",
    ]

    summary_word_hits = sum(1 for word in summary_words if word in lower_text)

    # Muitos pontinhos geralmente indicam linha de sumário.
    if dot_count >= 8 or long_dot_count >= 3:
        return True

    # Página com várias palavras típicas de sumário.
    if summary_word_hits >= 2 and dot_count >= 3:
        return True

    # Muitos capítulos em uma mesma página também indicam índice/sumário.
    if lower_text.count("capítulo") >= 3:
        return True

    return False


def calculate_relevance_score(text: str, search_term: str) -> int:
    """
    Calcula uma pontuação simples de relevância para uma página.

    A pontuação leva em conta:
    - quantas vezes o termo aparece;
    - se o termo aparece em títulos ou frases importantes;
    - se aparecem palavras que indicam explicação conceitual;
    - penalidade forte para páginas que parecem sumário.
    """

    lower_text = text.lower()
    lower_term = search_term.lower()

    occurrences = count_term_occurrences(text, search_term)

    if occurrences == 0:
        return 0

    score = occurrences

    # Padrões fortes: indicam que a página provavelmente ensina o conceito.
    strong_patterns = [
        f"conceito de {lower_term}",
        f"definição de {lower_term}",
        f"introdução às {lower_term}",
        f"introdução a {lower_term}",
        f"notação das {lower_term}",
        f"notação de {lower_term}",
        f"{lower_term} composta",
        f"{lower_term} inversa",
        "exemplo",
        "exemplos",
        "definida",
        "chama-se",
        "vamos considerar",
    ]

    for pattern in strong_patterns:
        if pattern in lower_text:
            score += 6

    # Palavras úteis em contexto de funções.
    concept_words = [
        "domínio",
        "imagem",
        "conjunto",
        "relação",
        "gráfico",
        "lei",
        "variável",
    ]

    for word in concept_words:
        if word in lower_text:
            score += 2

    # Penalidade forte para sumário/índice.
    if looks_like_summary_page(text):
        score -= 40

    # Penalidades menores.
    weak_patterns = [
        "prefácio",
        "apresentação",
    ]

    for pattern in weak_patterns:
        if pattern in lower_text:
            score -= 10

    return max(score, 0)


def search_text_in_pdf(
    pdf_path: str,
    search_term: str,
    max_results: int = 10,
    context_chars: int = 350,
):
    """
    Busca simples por termo dentro do PDF inteiro.

    Retorna as primeiras páginas encontradas.
    """

    document = open_pdf(pdf_path)
    results = []

    for page_index in range(len(document)):
        page = document[page_index]
        text = page.get_text()

        if search_term.lower() in text.lower():
            excerpt = create_text_excerpt(
                text=text,
                search_term=search_term,
                context_chars=context_chars,
            )

            results.append(
                {
                    "page": page_index + 1,
                    "term": search_term,
                    "excerpt": excerpt,
                }
            )

        if len(results) >= max_results:
            break

    document.close()

    return results


def search_relevant_pages(
    pdf_path: str,
    search_term: str,
    max_results: int = 5,
    context_chars: int = 500,
):
    """
    Busca páginas relevantes dentro do PDF.

    Diferente da busca simples, esta função:
    - percorre o PDF inteiro;
    - calcula pontuação por página;
    - ordena os resultados por relevância;
    - devolve os melhores trechos.
    """

    document = open_pdf(pdf_path)
    results = []

    for page_index in range(len(document)):
        page = document[page_index]
        text = page.get_text()

        score = calculate_relevance_score(
            text=text,
            search_term=search_term,
        )

        if score > 0:
            excerpt = create_text_excerpt(
                text=text,
                search_term=search_term,
                context_chars=context_chars,
            )

            results.append(
                {
                    "page": page_index + 1,
                    "term": search_term,
                    "score": score,
                    "excerpt": excerpt,
                }
            )

    document.close()

    results.sort(
        key=lambda result: result["score"],
        reverse=True,
    )

    return results[:max_results]


if __name__ == "__main__":
    """
    Teste manual do leitor de PDF.

    Para rodar diretamente:
    python3 tools/pdf_reader.py
    """

    pdf_path = "data/temp/fundamentos_matematica_elementar_001.pdf"

    print("Testando leitura do PDF...\n")

    total_pages = get_page_count(pdf_path)
    preview = extract_pdf_preview(pdf_path, page_number=1)

    print("PDF carregado com sucesso.")
    print(f"Total de páginas: {total_pages}")
    print("\nPrévia da página 1:\n")
    print(preview)

    print("\n" + "=" * 60)
    print("Testando busca por relevância sem sumário no topo...\n")

    search_term = "função"

    relevant_results = search_relevant_pages(
        pdf_path=pdf_path,
        search_term=search_term,
        max_results=5,
    )

    if not relevant_results:
        print(f"Nenhum resultado relevante encontrado para: {search_term}")
    else:
        print(f"Melhores resultados para: {search_term}\n")

        for result in relevant_results:
            print(f"Página {result['page']} | Relevância: {result['score']}")
            print(result["excerpt"])
            print("-" * 60)
