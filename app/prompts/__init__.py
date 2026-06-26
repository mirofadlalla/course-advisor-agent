from .identity import IDENTITY
from .behavior import BEHAVIOR
from .safety import SAFETY
from .formatting import FORMATTING
from .rag import RAG_PROMPT

SYSTEM_PROMPT = "\n\n".join(
    [
        IDENTITY,
        BEHAVIOR,
        SAFETY,
        FORMATTING,
        RAG_PROMPT,
    ]
)