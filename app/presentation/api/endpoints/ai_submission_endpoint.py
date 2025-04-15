from fastapi import APIRouter, HTTPException
from app.application.services.ai_submission_service import AISubmissionService

from app.presentation.api.models.submission_request import SubmissionRequest
from app.presentation.api.models.submission_response import (
    RetrievedDocument,
    SubmissionResponse,
)
from app.logs import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/darwin-chat-bot", tags=["AI Submission"])
submission_service = AISubmissionService()


@router.post("/ai-submission", response_model=SubmissionResponse)
async def process_submission(request: SubmissionRequest):
    """
    Endpoint to process AI message submissions.
    """
    logger.info(
        f"API: Receiving request to process submission: {request.input_message[:50]}..."
    )

    try:
        # Initialize config if not present
        if not submission_service.graph_builder.config.get("configurable"):
            submission_service.graph_builder.config["configurable"] = {}
            
        # Handle thread_id directly from request - maintain nested structure
        if request.thread_id:
            submission_service.graph_builder.config["configurable"]["thread_id"] = request.thread_id
        
        # Rest of the function remains the same
        responses = submission_service.process_message(request.input_message)
        
        # Rest of the function remains the same
        ai_content = ""
        retrieved_docs = []

        for msg in responses:
            if msg.type == "ai":
                ai_content = msg.content
            elif msg.type == "tool":
                try:
                    doc_content = msg.content
                    doc_metadata = msg.additional_kwargs.get("metadata", {})
                    retrieved_docs.append(
                        RetrievedDocument(content=doc_content, metadata=doc_metadata)
                    )
                except Exception as e:
                    logger.error(f"Error parsing tool message: {e}")

        thread_id = submission_service.graph_builder.config.get("configurable", {}).get(
            "thread_id"
        )

        return SubmissionResponse(
            content=ai_content,
            retrieved_docs=retrieved_docs if retrieved_docs else None,
            thread_id=thread_id,
        )

    except Exception as e:
        logger.error(f"Error processing submission: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error processing submission: {str(e)}"
        )
