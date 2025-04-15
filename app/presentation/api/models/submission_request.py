from pydantic import BaseModel, Field, UUID4
from typing import Optional, Dict, Any


class SubmissionRequest(BaseModel):
    input_message: str
    thread_id: Optional[UUID4] = Field(
        default=None, 
        description="Conversation thread ID (UUID4 format). If not provided, a new one will be generated."
    )