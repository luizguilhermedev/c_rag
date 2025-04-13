from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class Embedding:
    """Represents an embedding for a chunk of text."""

    vector: List[float]
    chunk_id: Optional[str] = None
    text: Optional[str] = None  # Adicionando o campo text

    def to_dict(self) -> dict:
        """
        Converte o objeto Embedding em um dicionário.

        Returns:
            dict: Representação do embedding como dicionário.
        """
        return asdict(self)
