from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, UUID4


class RetrievedDocument(BaseModel):
    """Model representing a retrieved document."""

    content: str
    metadata: dict = {}


class SubmissionResponse(BaseModel):
    """Model representing the response from an AI submission."""

    content: str
    retrieved_docs: Optional[List[RetrievedDocument]] = None
    thread_id: UUID4 = Field(
        description="The conversation thread ID to use in subsequent requests"
    )

