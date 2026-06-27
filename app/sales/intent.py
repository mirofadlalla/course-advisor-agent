"""Visitor intent detection from message text and conversation history."""

from __future__ import annotations

import re
from enum import Enum


class VisitorIntent(str, Enum):
    BROWSING = "browsing"
    COMPARING = "comparing"
    HESITANT = "hesitant"
    PRICE_SENSITIVE = "price_sensitive"
    READY_TO_ENROLL = "ready_to_enroll"


_ENROLL_PATTERNS = re.compile(
    r"\b(enroll|register|sign up|signup|subscribe|join|how do i (buy|pay|start)|"
    r"ابغى|أبغى|عايز|عاوز|سجل|التسجيل|اشترك|اشتراك|كيف (اسجل|أسجل|ادفع|أدفع))\b",
    re.IGNORECASE,
)
_COMPARE_PATTERNS = re.compile(
    r"\b(compare|versus|vs\.?|difference|better|which one|or the|"
    r"أيهما|ايهما|الفرق|مقارنة|ولا|أو ال|او ال)\b",
    re.IGNORECASE,
)
_HESITANT_PATTERNS = re.compile(
    r"\b(not sure|maybe later|think about|hesitant|worried|concern|"
    r"مش متأكد|محتار|بفكر|خايف|قلق)\b",
    re.IGNORECASE,
)
_PRICE_PATTERNS = re.compile(
    r"\b(price|cost|expensive|cheap|discount|payment|installment|budget|"
    r"سعر|أسعار|غالي|رخيص|خصم|دفع|قسط|ميزانية)\b",
    re.IGNORECASE,
)
_BROWSE_PATTERNS = re.compile(
    r"\b(what (courses|options|do you have)|tell me about|show me|explore|"
    r"ايه عندكم|إيه عندكم|عندكم ايه|شو عندكم|ايش عندكم)\b",
    re.IGNORECASE,
)


def _history_text(history: list[str]) -> str:
    return " ".join(history).lower()


def detect_intent(message: str, history: list[str] | None = None) -> VisitorIntent:
    """Classify visitor intent using message + prior user turns."""
    text = message.strip()
    combined = f"{_history_text(history or [])} {text.lower()}"

    if _ENROLL_PATTERNS.search(combined):
        return VisitorIntent.READY_TO_ENROLL
    if _COMPARE_PATTERNS.search(combined):
        return VisitorIntent.COMPARING
    if _PRICE_PATTERNS.search(combined):
        return VisitorIntent.PRICE_SENSITIVE
    if _HESITANT_PATTERNS.search(combined):
        return VisitorIntent.HESITANT
    if _BROWSE_PATTERNS.search(text.lower()):
        return VisitorIntent.BROWSING

    return VisitorIntent.BROWSING


_INTENT_GUIDANCE: dict[VisitorIntent, str] = {
    VisitorIntent.BROWSING: (
        "Visitor intent: BROWSING. Give a concise overview of relevant options. "
        "Mention free content when appropriate. Ask one clarifying question about "
        "their goal if needed."
    ),
    VisitorIntent.COMPARING: (
        "Visitor intent: COMPARING. Retrieve details for each option with tools, "
        "then compare duration, level, price, and outcomes in a clear table or "
        "bullets. Stay neutral — help them decide."
    ),
    VisitorIntent.HESITANT: (
        "Visitor intent: HESITANT. Acknowledge concerns. Answer with KB facts. "
        "Do not pressure. Offer to clarify one specific worry."
    ),
    VisitorIntent.PRICE_SENSITIVE: (
        "Visitor intent: PRICE-SENSITIVE. Retrieve exact prices from tools. "
        "Highlight free content or smaller paid options first. Never invent discounts."
    ),
    VisitorIntent.READY_TO_ENROLL: (
        "Visitor intent: READY TO ENROLL. Confirm the best-fit product from KB, "
        "summarize key facts, and give a clear enrollment next step (link or "
        "contact). Be helpful and direct, not pushy."
    ),
}


def build_intent_instructions(intent: VisitorIntent) -> str:
    """Per-turn dynamic instructions injected into the agent run."""
    return _INTENT_GUIDANCE[intent]
