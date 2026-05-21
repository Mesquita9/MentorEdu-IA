"""
Motor de busca da biblioteca do Thux.

Este arquivo junta:
- query_builder.py: entende a pergunta natural;
- drive_reader.py: encontra e baixa PDFs do Google Drive;
- pdf_reader.py: busca páginas relevantes dentro dos PDFs.

Função desta versão:
- receber uma pergunta natural;
- detectar disciplina e nível;
- gerar termos de busca;
- buscar esses termos na biblioteca;
- dar mais peso aos termos mais específicos;
- reduzir termos genéricos quando os específicos já encontraram bons resultados;
- devolver trechos com disciplina, nível, arquivo, página e relevância.

Esse é o começo do RAG do Thux:
pergunta -> biblioteca própria -> trechos -> resposta didática.
"""

from drive_reader import map_library, download_drive_file
from pdf_reader import search_relevant_pages
from query_builder import build_query_plan


def get_term_weight(term_position: int) -> int:
    """
    Define o peso de cada termo de busca.

    A ideia:
    - os primeiros termos gerados pelo query_builder são mais específicos;
    - termos finais costumam ser mais genéricos;
    - por isso, os primeiros devem pesar mais no ranking final.
    """

    weights = {
        0: 5,
        1: 4,
        2: 3,
        3: 1,
    }

    return weights.get(term_position, 1)


def is_generic_term(term: str) -> bool:
    """
    Identifica termos genéricos.

    Exemplo:
    - "função" é genérico.
    - "conceito de função" é específico.
    - "definição de função" é específico.

    Termos genéricos são úteis como backup, mas não devem dominar
    a resposta quando termos específicos já encontraram bons trechos.
    """

    generic_terms = [
        "função",
        "conjunto",
        "velocidade",
        "força",
        "energia",
        "aceleração",
    ]

    return term.lower().strip() in generic_terms


def has_good_specific_results(results, minimum_score: int = 30) -> bool:
    """
    Verifica se já existem bons resultados vindos de termos específicos.

    Se já existem bons resultados específicos, os termos genéricos
    podem ser reduzidos ou ignorados no ranking final.
    """

    for result in results:
        if not result["is_generic_term"] and result["weighted_score"] >= minimum_score:
            return True

    return False


def search_knowledge_base(
    question: str,
    max_files: int = 3,
    max_results_per_term: int = 3,
):
    """
    Busca conhecimento na biblioteca do Thux a partir de uma pergunta natural.

    Parâmetros:
    - question: pergunta feita pelo usuário.
    - max_files: quantidade máxima de PDFs analisados.
    - max_results_per_term: quantidade máxima de trechos por termo de busca.

    Retorno:
    - plano de busca;
    - lista de resultados encontrados.
    """

    # Cria o plano de busca a partir da pergunta.
    query_plan = build_query_plan(question)

    discipline_filter = query_plan["discipline"]
    level_filter = query_plan["level"]
    search_terms = query_plan["search_terms"]

    print("Plano de busca criado:\n")
    print(f"Pergunta: {query_plan['question']}")
    print(f"Disciplina: {discipline_filter}")
    print(f"Nível: {level_filter}")
    print(f"Termos de busca: {search_terms}")

    # Mapeia os PDFs disponíveis na biblioteca do Google Drive.
    library_items = map_library()

    # Filtra os PDFs pela disciplina e pelo nível detectados.
    filtered_items = []

    for item in library_items:
        if discipline_filter and item["discipline"] != discipline_filter:
            continue

        if level_filter and item["level"] != level_filter:
            continue

        filtered_items.append(item)

    # Se o filtro por nível for restritivo demais e não achar nada,
    # tenta novamente só com disciplina.
    if not filtered_items and level_filter:
        for item in library_items:
            if discipline_filter and item["discipline"] != discipline_filter:
                continue

            filtered_items.append(item)

    # Limita quantos PDFs serão analisados neste protótipo.
    filtered_items = filtered_items[:max_files]

    all_results = []

    for item in filtered_items:
        print(f"\nAnalisando: {item['name']} ({item['discipline']} / {item['level']})")

        downloaded_path = download_drive_file(
            file_id=item["id"],
            file_name=item["name"],
        )

        for term_position, term in enumerate(search_terms):
            term_weight = get_term_weight(term_position)
            generic = is_generic_term(term)

            print(f"Buscando termo: {term} | Peso: {term_weight} | Genérico: {generic}")

            page_results = search_relevant_pages(
                pdf_path=downloaded_path,
                search_term=term,
                max_results=max_results_per_term,
            )

            for result in page_results:
                weighted_score = result["score"] * term_weight

                all_results.append(
                    {
                        "discipline": item["discipline"],
                        "level": item["level"],
                        "file_name": item["name"],
                        "file_id": item["id"],
                        "page": result["page"],
                        "score": result["score"],
                        "term_weight": term_weight,
                        "weighted_score": weighted_score,
                        "term": term,
                        "is_generic_term": generic,
                        "excerpt": result["excerpt"],
                    }
                )

    # Remove resultados duplicados por arquivo + página + trecho.
    unique_results = []
    seen = set()

    for result in all_results:
        key = (
            result["file_id"],
            result["page"],
            result["excerpt"][:120],
        )

        if key in seen:
            continue

        seen.add(key)
        unique_results.append(result)

    # Se já existem bons resultados específicos,
    # reduzimos ou descartamos os genéricos.
    good_specific_results_found = has_good_specific_results(unique_results)

    if good_specific_results_found:
        specific_results = [
            result for result in unique_results
            if not result["is_generic_term"]
        ]

        generic_results = [
            result for result in unique_results
            if result["is_generic_term"]
        ]

        # Se já temos resultados específicos suficientes,
        # descartamos os genéricos para evitar misturar assunto avançado.
        if len(specific_results) >= 3:
            unique_results = specific_results
        else:
            for result in generic_results:
                result["weighted_score"] = int(result["weighted_score"] * 0.25)

            unique_results = specific_results + generic_results

    # Ordena pelo score ponderado final.
    unique_results.sort(
        key=lambda result: result["weighted_score"],
        reverse=True,
    )

    return {
        "query_plan": query_plan,
        "results": unique_results,
    }


def print_search_results(search_output, max_results: int = 8):
    """
    Mostra os resultados no terminal de forma legível.
    """

    query_plan = search_output["query_plan"]
    results = search_output["results"]

    print("\n" + "=" * 80)
    print("Resultado final da busca")
    print("=" * 80)

    print(f"\nPergunta original: {query_plan['question']}")
    print(f"Disciplina detectada: {query_plan['discipline']}")
    print(f"Nível detectado: {query_plan['level']}")
    print(f"Termos usados: {query_plan['search_terms']}")

    if not results:
        print("\nNenhum resultado encontrado na biblioteca.")
        return

    print("\nTrechos encontrados:\n")

    for result in results[:max_results]:
        print(
            f"Disciplina: {result['discipline']} | "
            f"Nível: {result['level']} | "
            f"Arquivo: {result['file_name']} | "
            f"Página: {result['page']} | "
            f"Termo: {result['term']} | "
            f"Genérico: {result['is_generic_term']} | "
            f"Relevância bruta: {result['score']} | "
            f"Peso do termo: {result['term_weight']} | "
            f"Relevância final: {result['weighted_score']}"
        )

        print(result["excerpt"])
        print("-" * 80)


if __name__ == "__main__":
    """
    Teste manual.

    Para rodar:
    python3 tools/knowledge_search.py
    """

    question = "T, não entendi o que é função. Me explica como se fosse ensino médio."

    search_output = search_knowledge_base(
        question=question,
        max_files=1,
        max_results_per_term=3,
    )

    print_search_results(search_output)
