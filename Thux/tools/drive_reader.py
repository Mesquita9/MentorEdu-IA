"""
Leitor do Google Drive para o Thux.

Este arquivo permite que o Thux acesse a biblioteca privada no Google Drive
usando uma conta de serviço.

Funciona em dois modos:

1. Ambiente local:
   - usa o arquivo credentials/google_drive_credentials.json

2. Ambiente em nuvem, como Render:
   - usa a variável de ambiente GOOGLE_DRIVE_CREDENTIALS_JSON
   - essa variável deve conter o conteúdo completo do JSON da conta de serviço

Funções:
- conectar ao Google Drive;
- encontrar a pasta principal Thux-AI;
- mapear disciplinas e níveis;
- listar PDFs organizados;
- baixar PDFs temporariamente para leitura local;
- testar a leitura do PDF baixado com o pdf_reader.py.
"""

import io
import os
import json

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from pdf_reader import get_page_count, extract_pdf_preview


# Caminho local da chave JSON da conta de serviço.
# Este arquivo NÃO deve ir para o GitHub.
CREDENTIALS_PATH = "credentials/google_drive_credentials.json"


# Nome da variável de ambiente usada no Render.
# Ela deve conter o JSON completo da conta de serviço.
GOOGLE_DRIVE_CREDENTIALS_ENV = "GOOGLE_DRIVE_CREDENTIALS_JSON"


# Permissão apenas de leitura no Google Drive.
SCOPES = (
    "https://www.googleapis.com/auth/drive.readonly",
)


# Nome da pasta principal no Google Drive.
ROOT_FOLDER_NAME = "Thux-AI"


# Disciplinas esperadas na biblioteca.
DISCIPLINES = (
    "Matemática",
    "Física",
)


# Níveis esperados dentro de cada disciplina.
LEVELS = (
    "Elementar",
    "Avançado",
)


# Pastas de exercícios e materiais práticos.
# Ainda está preparado para uso futuro.
MATERIAL_FOLDERS = (
    "Listas",
)


# Pasta local para downloads temporários.
TEMP_DIR = "data/temp"


def get_drive_credentials():
    """
    Carrega as credenciais do Google Drive.

    Prioridade:
    1. Se existir GOOGLE_DRIVE_CREDENTIALS_JSON, usa a variável de ambiente.
       Esse será o modo usado no Render.

    2. Se não existir variável de ambiente, usa o arquivo local:
       credentials/google_drive_credentials.json
    """

    credentials_json = os.getenv(GOOGLE_DRIVE_CREDENTIALS_ENV)

    if credentials_json:
        try:
            credentials_info = json.loads(credentials_json)

            credentials = service_account.Credentials.from_service_account_info(
                credentials_info,
                scopes=SCOPES,
            )

            return credentials

        except json.JSONDecodeError as error:
            raise ValueError(
                "A variável GOOGLE_DRIVE_CREDENTIALS_JSON existe, "
                "mas não contém um JSON válido."
            ) from error

    if not os.path.exists(CREDENTIALS_PATH):
        raise FileNotFoundError(
            "Credenciais do Google Drive não encontradas.\n"
            f"Modo local esperado: {CREDENTIALS_PATH}\n"
            f"Modo nuvem esperado: variável {GOOGLE_DRIVE_CREDENTIALS_ENV}"
        )

    credentials = service_account.Credentials.from_service_account_file(
        CREDENTIALS_PATH,
        scopes=SCOPES,
    )

    return credentials


def get_drive_service():
    """
    Cria uma conexão autenticada com o Google Drive.
    """

    credentials = get_drive_credentials()

    service = build(
        "drive",
        "v3",
        credentials=credentials,
    )

    return service


def search_folder_by_name(service, folder_name: str, parent_id: str | None = None):
    """
    Procura uma pasta pelo nome.

    Se parent_id for informado, procura apenas dentro daquela pasta.
    """

    query_parts = [
        "mimeType = 'application/vnd.google-apps.folder'",
        f"name = '{folder_name}'",
        "trashed = false",
    ]

    if parent_id:
        query_parts.append(f"'{parent_id}' in parents")

    query = " and ".join(query_parts)

    results = service.files().list(
        q=query,
        fields="files(id, name, mimeType)",
        pageSize=10,
    ).execute()

    folders = results.get("files", [])

    if not folders:
        return None

    return folders[0]


def list_pdfs_in_folder(service, folder_id: str):
    """
    Lista arquivos PDF dentro de uma pasta específica.
    """

    query = (
        f"'{folder_id}' in parents and "
        "mimeType = 'application/pdf' and "
        "trashed = false"
    )

    results = service.files().list(
        q=query,
        fields="files(id, name, mimeType)",
        pageSize=100,
    ).execute()

    return results.get("files", [])


def map_library():
    """
    Mapeia a biblioteca do Google Drive.

    Retorna uma lista de materiais encontrados com:
    - disciplina;
    - nível;
    - nome do arquivo;
    - id do arquivo no Drive.
    """

    service = get_drive_service()

    root_folder = search_folder_by_name(service, ROOT_FOLDER_NAME)

    if not root_folder:
        raise FileNotFoundError(
            f"Pasta principal '{ROOT_FOLDER_NAME}' não encontrada no Google Drive."
        )

    library_items = []

    for discipline in DISCIPLINES:
        discipline_folder = search_folder_by_name(
            service,
            discipline,
            parent_id=root_folder["id"],
        )

        if not discipline_folder:
            continue

        for level in LEVELS:
            level_folder = search_folder_by_name(
                service,
                level,
                parent_id=discipline_folder["id"],
            )

            if not level_folder:
                continue

            pdfs = list_pdfs_in_folder(service, level_folder["id"])

            for pdf in pdfs:
                library_items.append(
                    {
                        "discipline": discipline,
                        "level": level,
                        "name": pdf["name"],
                        "id": pdf["id"],
                    }
                )

    return library_items


def download_drive_file(file_id: str, file_name: str, output_dir: str = TEMP_DIR) -> str:
    """
    Baixa um arquivo do Google Drive para uma pasta temporária local.

    Parâmetros:
    - file_id: ID do arquivo no Google Drive;
    - file_name: nome do arquivo para salvar localmente;
    - output_dir: pasta onde o arquivo será salvo.

    Retorno:
    - caminho local do arquivo baixado.
    """

    service = get_drive_service()

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, file_name)

    request = service.files().get_media(fileId=file_id)

    with io.FileIO(output_path, "wb") as downloaded_file:
        downloader = MediaIoBaseDownload(downloaded_file, request)

        done = False

        while not done:
            status, done = downloader.next_chunk()

            if status:
                progress = int(status.progress() * 100)
                print(f"Download em andamento: {progress}%")

    return output_path


def test_first_pdf_from_library():
    """
    Testa o fluxo completo:

    Google Drive
    → encontra biblioteca
    → acha o primeiro PDF
    → baixa para data/temp/
    → lê com pdf_reader.py
    → mostra total de páginas e prévia da página 1.
    """

    items = map_library()

    print("Biblioteca encontrada no Google Drive:\n")

    if not items:
        print("Nenhum PDF encontrado na estrutura esperada.")
        return

    for item in items:
        print(
            f"- Disciplina: {item['discipline']} | "
            f"Nível: {item['level']} | "
            f"Arquivo: {item['name']} | "
            f"ID: {item['id']}"
        )

    print("\nBaixando o primeiro PDF encontrado para teste...\n")

    first_item = items[0]

    downloaded_path = download_drive_file(
        file_id=first_item["id"],
        file_name=first_item["name"],
    )

    print("\nPDF baixado com sucesso em:")
    print(downloaded_path)

    print("\nTestando leitura do PDF baixado...\n")

    total_pages = get_page_count(downloaded_path)
    preview = extract_pdf_preview(downloaded_path, page_number=1)

    print(f"Total de páginas: {total_pages}")
    print("\nPrévia da página 1:\n")
    print(preview)


if __name__ == "__main__":
    """
    Teste manual.

    Para rodar:
    python3 tools/drive_reader.py
    """

    test_first_pdf_from_library()
