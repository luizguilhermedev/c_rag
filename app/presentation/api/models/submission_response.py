from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class RetrievedDocument(BaseModel):
    """Model representing a retrieved document."""

    content: str
    metadata: Dict[str, Any]


class SubmissionResponse(BaseModel):
    """Model representing the response from an AI submission."""

    content: str
    retrieved_docs: Optional[List[RetrievedDocument]] = None

