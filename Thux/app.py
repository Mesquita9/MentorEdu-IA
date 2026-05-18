from fastapi import FastAPI

app = FastAPI(
    title="Thux-AI",
    description="Assistente pessoal didático focado em Física, Matemática e apoio docente.",
    version="0.1.0"
)


@app.get("/")
def home():
    return {
        "status": "Thux online",
        "message": "O núcleo inicial do Thux-AI está funcionando."
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Thux-AI"
    }
