"""
Configurações centrais do Thux-AI.

Este arquivo guarda informações gerais do projeto.
Não coloque chaves de API, senhas ou tokens diretamente aqui.
Segredos devem ficar em variáveis de ambiente.
"""

PROJECT_NAME = "Thux-AI"
ASSISTANT_NAME = "Thux"

ASSISTANT_ALIASES = [
    "T",
    "Pinguim",
]

PROJECT_DESCRIPTION = (
    "Assistente pessoal didático focado em Física, Matemática, "
    "Química como apoio e futura rotina docente."
)

# Provedor de IA usado inicialmente
AI_PROVIDER = "groq"

# Modelo inicial para o protótipo
AI_MODEL = "llama-3.1-8b-instant"

# Nome da variável de ambiente onde ficará a chave da Groq
GROQ_API_KEY_ENV = "GROQ_API_KEY"

# Disciplinas principais do projeto
MAIN_DISCIPLINES = [
    "matematica",
    "fisica",
    "quimica",
]

# Disciplina prioritária do núcleo do Thux
CORE_DISCIPLINES = [
    "matematica",
    "fisica",
]

# Configurações gerais de resposta
DEFAULT_LANGUAGE = "pt-BR"
MAX_RESPONSE_TOKENS = 1200
TEMPERATURE = 0.7

# Pastas principais do projeto
DATA_DIR = "data"
KNOWLEDGE_DIR = "data/knowledge"
CONVERSATIONS_DIR = "data/conversations"
LOGS_DIR = "data/logs"

# Status inicial do projeto
PROJECT_STATUS = "prototype"
