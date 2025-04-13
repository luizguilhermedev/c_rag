from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class Embedding:
    """Represents an embedding for a chunk of text."""

    vector: List[float]
    chunk_id: Optional[str] = None
    text: Optional[str] = None

    def to_dict(self) -> dict:
        """
        Converts the Embedding object into a dictionary.

        Returns:
            dict: Representation of the embedding as a dictionary.
        """
        return asdict(self)
