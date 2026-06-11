"""
Rotas de geração de gráficos do Thux-AI.

Recebe dados do frontend e chama o motor de gráficos em tools/graph_generator.py.
"""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from tools.graph_generator import generate_graph


router = APIRouter()


class GraphPoint(BaseModel):
    x: float = Field(..., example=1)
    y: float = Field(..., example=2)


class GraphRequest(BaseModel):
    graph_type: str = Field(
        ...,
        example="quadratica",
        description="Tipos aceitos: afim, quadratica, trigonometrica, exponencial, tabela",
    )

    expression: str | None = Field(
        None,
        example="x**2 - 5*x + 6",
        description="Expressão em função de x. Exemplo: 2*x + 1, x**2 - 5*x + 6, sin(x), 2**x",
    )

    x_min: float = Field(
        -10,
        example=-2,
        description="Valor mínimo de x no gráfico.",
    )

    x_max: float = Field(
        10,
        example=6,
        description="Valor máximo de x no gráfico.",
    )

    points: list[GraphPoint] | None = Field(
        None,
        description="Lista de pontos para gráfico por tabela.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "graph_type": "quadratica",
                "expression": "x**2 - 5*x + 6",
                "x_min": -2,
                "x_max": 6,
                "points": None,
            }
        }


@router.post("/api/graph")
def create_graph(data: GraphRequest) -> dict[str, Any]:
    """
    Gera um gráfico a partir dos dados enviados pelo frontend.
    """

    try:
        result = generate_graph(
            graph_type=data.graph_type,
            expression=data.expression,
            x_min=data.x_min,
            x_max=data.x_max,
            points=[point.model_dump() for point in data.points] if data.points else None,
        )

        return {
            "ok": True,
            "graph": result,
        }

    except Exception as error:
        return {
            "ok": False,
            "error": str(error),
        }


@router.post("/graph")
def create_graph_alias(data: GraphRequest) -> dict[str, Any]:
    """
    Alias para testes rápidos.
    """

    return create_graph(data)
