from .behavior import BEHAVIOR
from .formatting import FORMATTING
from .identity import IDENTITY
from .rag import RAG_PROMPT
from .safety import SAFETY
from .sales import SALES

SYSTEM_PROMPT = "\n\n".join(
    [
        IDENTITY,
        BEHAVIOR,
        SALES,
        SAFETY,
        FORMATTING,
        RAG_PROMPT,
    ]
)
