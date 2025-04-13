from dataclasses import dataclass
from typing import Optional, Dict, Any
from uuid import uuid4


@dataclass
class Chunk:
    """
    This class represents a smaller part of a document, which can be processed
    """

    text: str
    id: str = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid4())

        if self.metadata is None:
            self.metadata = {}
