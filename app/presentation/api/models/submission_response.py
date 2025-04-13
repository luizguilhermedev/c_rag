from pydantic import BaseModel


class SubmissionResponse(BaseModel):
    content: str
