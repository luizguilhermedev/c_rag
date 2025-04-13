from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.presentation.api.endpoints.ai_submission_endpoint import (
    router as ai_submission_router,
)

API_PREFIX = "/api/v1"

app = FastAPI(
    title="Darwin Chat Bot API",
    description="API para processar submissões de mensagens de IA.",
    version="1.0.0",
)


@app.get("/", include_in_schema=False)
def redirect_to_docs():
    """
    Redireciona para a documentação interativa da API.
    """
    return RedirectResponse(url="/docs")


# Incluir o roteador de submissões de IA
app.include_router(ai_submission_router, prefix=API_PREFIX)
