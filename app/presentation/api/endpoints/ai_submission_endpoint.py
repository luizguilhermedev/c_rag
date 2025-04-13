import logging
from fastapi import APIRouter, HTTPException
from app.application.services.ai_submission_service import AISubmissionService
from langchain.schema import AIMessage

from app.presentation.api.models.submission_request import SubmissionRequest
from app.presentation.api.models.submission_response import SubmissionResponse

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/darwin-chat-bot", tags=["AI Submission"])
submission_service = AISubmissionService()


@router.post("/ai-submission", response_model=SubmissionResponse)
async def process_submission(request: SubmissionRequest):
    """
    Endpoint to process AI message submissions.
    """
    logger.info("Receiving request to process submission: %s", request.dict())
    try:
        # Ensure the config is in the expected format
        thread_id = request.config.get("thread_id") or request.config.get(
            "additionalProp1", {}
        ).get("thread_id")
        if not thread_id:
            raise KeyError("thread_id not found in the config.")

        config = {"configurable": {"thread_id": thread_id}}

        response: AIMessage = submission_service.process_submission(
            input_message=request.input_message, config=config
        )
        if response:
            logger.info(
                "Submission processed successfully. Response: %s", response.content
            )
            return SubmissionResponse(content=response.content)
        else:
            logger.warning("No AIMessage found for the submission.")
            raise HTTPException(status_code=404, detail="No AIMessage found.")
    except KeyError as e:
        logger.error("Error in config format: %s", str(e))
        raise HTTPException(
            status_code=400, detail=f"Error in config format: {str(e)}"
        )
    except Exception as e:
        logger.error("Error processing submission: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")
