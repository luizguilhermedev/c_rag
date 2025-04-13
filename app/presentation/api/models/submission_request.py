from pydantic import BaseModel


class SubmissionRequest(BaseModel):
    input_message: str
    config: dict
