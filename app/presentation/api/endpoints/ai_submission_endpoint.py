import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.application.services.ai_submission_service import AISubmissionService
from langchain.schema import AIMessage

from app.presentation.api.models.submission_request import SubmissionRequest
from app.presentation.api.models.submission_response import SubmissionResponse

# Configurar o logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/darwin-chat-bot", tags=["AI Submission"])
submission_service = AISubmissionService()


@router.post("/ai-submission", response_model=SubmissionResponse)
async def process_submission(request: SubmissionRequest):
    """
    Endpoint para processar submissões de mensagens de IA.
    """
    logger.info("Recebendo requisição para processar submissão: %s", request.dict())
    try:
        # Garantir que o config esteja no formato esperado
        thread_id = request.config.get("thread_id") or request.config.get(
            "additionalProp1", {}
        ).get("thread_id")
        if not thread_id:
            raise KeyError("thread_id não encontrado no config.")

        config = {"configurable": {"thread_id": thread_id}}

        response: AIMessage = submission_service.process_submission(
            input_message=request.input_message, config=config
        )
        if response:
            logger.info(
                "Submissão processada com sucesso. Resposta: %s", response.content
            )
            return SubmissionResponse(content=response.content)
        else:
            logger.warning("Nenhum AIMessage encontrado para a submissão.")
            raise HTTPException(status_code=404, detail="Nenhum AIMessage encontrado.")
    except KeyError as e:
        logger.error("Erro no formato do config: %s", str(e))
        raise HTTPException(
            status_code=400, detail=f"Erro no formato do config: {str(e)}"
        )
    except Exception as e:
        logger.error("Erro ao processar submissão: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Erro interno no servidor.")
