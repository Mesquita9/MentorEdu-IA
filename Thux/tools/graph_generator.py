"""
Gerador de gráficos do Thux-AI.

Versão inicial:
- função afim;
- função quadrática;
- trigonométrica simples;
- exponencial simples;
- tabela de pontos.

Retorna imagem em base64 para o frontend exibir.
"""

import base64
import io
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def figure_to_base64() -> str:
    """
    Converte a figura atual do Matplotlib para base64.
    """

    buffer = io.BytesIO()

    plt.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        dpi=150,
    )

    buffer.seek(0)

    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    plt.close()

    return f"data:image/png;base64,{image_base64}"


def normalize_expression(expression: str) -> str:
    """
    Ajusta expressões digitadas pelo usuário para o padrão aceito pelo Python/SymPy.
    """

    if not expression:
        raise ValueError("Nenhuma expressão foi enviada.")

    normalized = expression.strip()

    replacements = {
        "^": "**",
        "×": "*",
        "sen": "sin",
        "tg": "tan",
        "π": "pi",
        "e^": "exp",
    }

    for old, new in replacements.items():
        normalized = normalized.replace(old, new)

    return normalized


def parse_expression(expression: str):
    """
    Converte texto em expressão SymPy.
    """

    x = sp.Symbol("x")
    normalized = normalize_expression(expression)

    allowed_symbols = {
        "x": x,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "sqrt": sp.sqrt,
        "log": sp.log,
        "pi": sp.pi,
        "E": sp.E,
    }

    expr = sp.sympify(normalized, locals=allowed_symbols)

    return x, expr


def safe_float(value: Any) -> float | None:
    """
    Tenta converter um valor simbólico em float.
    """

    try:
        return float(value)
    except Exception:
        return None


def analyze_expression(expression: str, graph_type: str) -> tuple[Any, dict]:
    """
    Faz uma análise matemática inicial da função.
    """

    x, expr = parse_expression(expression)
    expanded = sp.expand(expr)

    analysis = {
        "expressao": str(expr),
        "tipo": graph_type,
    }

    try:
        roots = sp.solve(expr, x)
        real_roots = []

        for root in roots:
            root_float = safe_float(root)

            if root_float is not None:
                real_roots.append(str(sp.simplify(root)))

        analysis["raizes"] = real_roots

    except Exception:
        analysis["raizes"] = []

    try:
        y_intercept = expr.subs(x, 0)
        analysis["intercepto_y"] = str(sp.simplify(y_intercept))
    except Exception:
        analysis["intercepto_y"] = None

    if graph_type == "afim":
        try:
            a = expanded.coeff(x, 1)
            b = expanded.coeff(x, 0)

            analysis["a"] = str(sp.simplify(a))
            analysis["b"] = str(sp.simplify(b))
            analysis["coeficiente_angular"] = str(sp.simplify(a))
            analysis["coeficiente_linear"] = str(sp.simplify(b))

            if a > 0:
                analysis["comportamento"] = "crescente"
            elif a < 0:
                analysis["comportamento"] = "decrescente"
            else:
                analysis["comportamento"] = "constante"

        except Exception:
            analysis["aviso"] = "Não consegui calcular todos os dados da função afim."

    if graph_type == "quadratica":
        try:
            a = expanded.coeff(x, 2)
            b = expanded.coeff(x, 1)
            c = expanded.coeff(x, 0)

            if a == 0:
                raise ValueError("A expressão não parece ser quadrática, pois a = 0.")

            delta = b**2 - 4 * a * c
            xv = -b / (2 * a)
            yv = expr.subs(x, xv)

            analysis["a"] = str(sp.simplify(a))
            analysis["b"] = str(sp.simplify(b))
            analysis["c"] = str(sp.simplify(c))
            analysis["delta"] = str(sp.simplify(delta))
            analysis["vertice"] = {
                "x": str(sp.simplify(xv)),
                "y": str(sp.simplify(yv)),
            }
            analysis["concavidade"] = "para cima" if a > 0 else "para baixo"

        except Exception as error:
            analysis["aviso"] = f"Não consegui calcular todos os dados da quadrática: {error}"

    if graph_type == "trigonometrica":
        analysis["observacao"] = (
            "Análise trigonométrica inicial. Futuramente calcularemos amplitude, período, fase e deslocamento."
        )

    if graph_type == "exponencial":
        analysis["observacao"] = (
            "Análise exponencial inicial. Futuramente calcularemos crescimento/decaimento e assíntota."
        )

    return expr, analysis


def prepare_y_values(function, x_values):
    """
    Calcula y e remove valores inválidos.
    """

    y_values = function(x_values)

    if np.isscalar(y_values):
        y_values = np.full_like(x_values, y_values, dtype=float)

    y_values = np.array(y_values, dtype=float)

    valid_mask = np.isfinite(y_values)

    return x_values[valid_mask], y_values[valid_mask]


def add_cartesian_plane():
    """
    Adiciona eixos cartesianos e grade.
    """

    plt.axhline(0, linewidth=1.2)
    plt.axvline(0, linewidth=1.2)
    plt.grid(True, alpha=0.35)


def mark_roots(expr, x_symbol, x_min: float, x_max: float):
    """
    Marca raízes reais no gráfico.
    """

    try:
        roots = sp.solve(expr, x_symbol)

        for root in roots:
            root_float = safe_float(root)

            if root_float is None:
                continue

            if x_min <= root_float <= x_max:
                plt.scatter([root_float], [0], s=55)
                plt.annotate(
                    f"({root_float:.2f}, 0)",
                    (root_float, 0),
                    textcoords="offset points",
                    xytext=(6, 8),
                )

    except Exception:
        pass


def mark_y_intercept(expr, x_symbol, x_min: float, x_max: float):
    """
    Marca o intercepto no eixo y.
    """

    try:
        y_intercept = safe_float(expr.subs(x_symbol, 0))

        if y_intercept is not None and x_min <= 0 <= x_max:
            plt.scatter([0], [y_intercept], s=55)
            plt.annotate(
                f"(0, {y_intercept:.2f})",
                (0, y_intercept),
                textcoords="offset points",
                xytext=(6, 8),
            )

    except Exception:
        pass


def mark_quadratic_vertex(expr, x_symbol, x_min: float, x_max: float):
    """
    Marca o vértice da parábola.
    """

    try:
        expanded = sp.expand(expr)
        a = expanded.coeff(x_symbol, 2)
        b = expanded.coeff(x_symbol, 1)

        if a == 0:
            return

        xv = safe_float(-b / (2 * a))

        if xv is None:
            return

        yv = safe_float(expr.subs(x_symbol, xv))

        if yv is None:
            return

        if x_min <= xv <= x_max:
            plt.scatter([xv], [yv], s=70)
            plt.annotate(
                f"V({xv:.2f}, {yv:.2f})",
                (xv, yv),
                textcoords="offset points",
                xytext=(6, -16),
            )

    except Exception:
        pass


def generate_expression_graph(
    graph_type: str,
    expression: str,
    x_min: float,
    x_max: float,
) -> dict:
    """
    Gera gráfico de uma expressão matemática.
    """

    if x_min >= x_max:
        raise ValueError("x_min precisa ser menor que x_max.")

    x_symbol, expr = parse_expression(expression)
    _, analysis = analyze_expression(expression, graph_type)

    function = sp.lambdify(x_symbol, expr, modules=["numpy"])

    x_values = np.linspace(x_min, x_max, 800)
    x_values, y_values = prepare_y_values(function, x_values)

    if len(x_values) == 0:
        raise ValueError("Não consegui gerar valores válidos para essa expressão.")

    plt.figure(figsize=(8, 5))
    add_cartesian_plane()

    plt.plot(x_values, y_values, linewidth=2.2)

    mark_roots(expr, x_symbol, x_min, x_max)
    mark_y_intercept(expr, x_symbol, x_min, x_max)

    if graph_type == "quadratica":
        mark_quadratic_vertex(expr, x_symbol, x_min, x_max)

    plt.title(f"f(x) = {expression}")
    plt.xlabel("x")
    plt.ylabel("f(x)")

    image = figure_to_base64()

    return {
        "image": image,
        "analysis": analysis,
    }


def generate_points_graph(points: list[dict]) -> dict:
    """
    Gera gráfico a partir de tabela de pontos.
    """

    if not points:
        raise ValueError("Nenhum ponto foi enviado.")

    x_values = [float(point["x"]) for point in points]
    y_values = [float(point["y"]) for point in points]

    plt.figure(figsize=(8, 5))
    add_cartesian_plane()

    plt.plot(x_values, y_values, linewidth=2)
    plt.scatter(x_values, y_values, s=60)

    for x_value, y_value in zip(x_values, y_values):
        plt.annotate(
            f"({x_value:g}, {y_value:g})",
            (x_value, y_value),
            textcoords="offset points",
            xytext=(6, 8),
        )

    plt.title("Gráfico por tabela de pontos")
    plt.xlabel("x")
    plt.ylabel("y")

    image = figure_to_base64()

    return {
        "image": image,
        "analysis": {
            "tipo": "tabela_pontos",
            "quantidade_pontos": len(points),
            "pontos": points,
        },
    }


def normalize_graph_type(graph_type: str) -> str:
    """
    Normaliza o tipo de gráfico.
    """

    normalized = graph_type.lower().strip()

    aliases = {
        "quadrática": "quadratica",
        "quadratico": "quadratica",
        "parabola": "quadratica",
        "parábola": "quadratica",
        "reta": "afim",
        "linear": "afim",
        "função afim": "afim",
        "funcao afim": "afim",
        "trigonométrica": "trigonometrica",
        "trigonometrica": "trigonometrica",
        "seno": "trigonometrica",
        "cosseno": "trigonometrica",
        "exponencial": "exponencial",
        "tabela": "tabela",
        "pontos": "tabela",
        "tabela_pontos": "tabela",
    }

    return aliases.get(normalized, normalized)


def generate_graph(
    graph_type: str,
    expression: str | None = None,
    x_min: float = -10,
    x_max: float = 10,
    points: list[dict] | None = None,
) -> dict:
    """
    Função principal chamada pela rota /api/graph.
    """

    normalized_type = normalize_graph_type(graph_type)

    if normalized_type == "tabela":
        return generate_points_graph(points or [])

    supported_types = {
        "afim",
        "quadratica",
        "trigonometrica",
        "exponencial",
    }

    if normalized_type not in supported_types:
        raise ValueError(
            "Tipo de gráfico não suportado. Use: afim, quadratica, trigonometrica, exponencial ou tabela."
        )

    return generate_expression_graph(
        graph_type=normalized_type,
        expression=expression or "",
        x_min=x_min,
        x_max=x_max,
    )
