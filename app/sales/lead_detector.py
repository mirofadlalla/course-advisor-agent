"""Contextual lead / buying-signal detection from conversation."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from app.sales.intent import VisitorIntent, detect_intent

ContactPreference = Literal["phone", "whatsapp", "email", "any"]


class LeadSignals(BaseModel):
    """Structured buying signals extracted from a conversation turn."""

    is_qualified: bool = False
    qualification_score: float = Field(default=0.0, ge=0.0, le=1.0)
    buying_signals: list[str] = Field(default_factory=list)
    objections: list[str] = Field(default_factory=list)
    products_mentioned: list[str] = Field(default_factory=list)
    goal: str = ""
    level: str = ""
    name: str = ""
    phone: str = ""
    whatsapp: str = ""
    email: str = ""
    city: str = ""
    language: str = ""
    dialect: str = ""
    contact_preference: ContactPreference = "any"
    sales_action: str = ""


_BUYING_SIGNAL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("asked_price", re.compile(r"\b(price|cost|how much|much|سعر|أسعار|كم)\b", re.I)),
    ("asked_payment", re.compile(r"\b(pay|payment|installment|دفع|قسط)\b", re.I)),
    ("asked_enrollment", re.compile(r"\b(enroll|register|sign up|سجل|التسجيل|اشترك)\b", re.I)),
    ("asked_start_date", re.compile(r"\b(start|when|begin|متى|يبدأ|بداية)\b", re.I)),
    ("asked_contact", re.compile(r"\b(call|whatsapp|email|اتصل|واتس|ايميل)\b", re.I)),
    ("ready_to_buy", re.compile(r"\b(want to (buy|enroll|join)|عايز|ابغى|أبغى)\b", re.I)),
]

_OBJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("price_objection", re.compile(r"\b(too expensive|غالي|expensive)\b", re.I)),
    ("time_objection", re.compile(r"\b(no time|busy|مش فاضي|مافي وقت)\b", re.I)),
    ("trust_objection", re.compile(r"\b(not sure|trust|scam|مش متأكد|نصب)\b", re.I)),
]

_PRODUCT_PATTERNS = re.compile(
    r"\b(python|fullstack|full stack|data science|cyber|pentest|pen test|soc|"
    r"diploma|bootcamp|track|ms sql|sql server|ai|machine learning)\b",
    re.IGNORECASE,
)

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"(?:\+?\d[\d\s\-()]{7,}\d)")
_WHATSAPP = re.compile(r"(?:whatsapp|واتس(?:اب)?)\s*[:\-]?\s*(\+?\d[\d\s\-]{7,}\d)?", re.I)
_NAME = re.compile(
    r"(?:my name is|i am|i'm|اسمي|انا|أنا)\s+([A-Za-z\u0600-\u06FF]{2,30})",
    re.IGNORECASE,
)
_CITY = re.compile(
    r"(?:from|in|live in|من|في)\s+([A-Za-z\u0600-\u06FF]{2,25})",
    re.IGNORECASE,
)

_LEVEL_PATTERNS = [
    (re.compile(r"\b(beginner|مبتدئ)\b", re.I), "beginner"),
    (re.compile(r"\b(intermediate|متوسط)\b", re.I), "intermediate"),
    (re.compile(r"\b(advanced|متقدم)\b", re.I), "advanced"),
]

_GOAL_PATTERNS = [
    (re.compile(r"\b(career|job|work|وظيفة|شغل)\b", re.I), "career change"),
    (re.compile(r"\b(learn|study|تعلم|دراسة)\b", re.I), "skill building"),
]


def _detect_language_dialect(text: str) -> tuple[str, str]:
    if not re.search(r"[\u0600-\u06FF]", text):
        return "en", ""
    if re.search(r"\b(ازاي|عايز|إزاي|كدا|كده|حاجة)\b", text):
        return "ar", "egyptian"
    if re.search(r"\b(وش|ابغى|أبغى|كيف|حلو|زين)\b", text):
        return "ar", "saudi"
    if re.search(r"\b(شو|بدي|كتير|منيح)\b", text):
        return "ar", "syrian"
    return "ar", "msa"


def detect_lead_signals(
    message: str,
    history: list[str] | None = None,
    assistant_reply: str = "",
) -> LeadSignals:
    """Detect buying signals and contact info from message + history."""
    history = history or []
    combined = " ".join([*history, message, assistant_reply])
    intent = detect_intent(message, history)

    signals = LeadSignals()
    for label, pattern in _BUYING_SIGNAL_PATTERNS:
        if pattern.search(combined):
            signals.buying_signals.append(label)

    for label, pattern in _OBJECTION_PATTERNS:
        if pattern.search(combined):
            signals.objections.append(label)

    products = {m.group(0).strip() for m in _PRODUCT_PATTERNS.finditer(combined)}
    signals.products_mentioned = sorted(products)

    for pattern, level in _LEVEL_PATTERNS:
        if pattern.search(combined):
            signals.level = level
            break

    for pattern, goal in _GOAL_PATTERNS:
        if pattern.search(combined):
            signals.goal = goal
            break

    if email := _EMAIL.search(message):
        signals.email = email.group(0)
    if phone := _PHONE.search(message):
        signals.phone = phone.group(0).strip()
    if wa := _WHATSAPP.search(message):
        signals.whatsapp = (wa.group(1) or signals.phone).strip()
    if name := _NAME.search(message):
        signals.name = name.group(1).strip()
    if city := _CITY.search(message):
        signals.city = city.group(1).strip()

    lang, dialect = _detect_language_dialect(message)
    signals.language = lang
    signals.dialect = dialect

    if re.search(r"\b(whatsapp|واتس)\b", message, re.I):
        signals.contact_preference = "whatsapp"
    elif signals.email:
        signals.contact_preference = "email"
    elif signals.phone:
        signals.contact_preference = "phone"

    score = min(1.0, len(signals.buying_signals) * 0.2)
    if intent == VisitorIntent.READY_TO_ENROLL:
        score = min(1.0, score + 0.35)
    if signals.email or signals.phone or signals.whatsapp:
        score = min(1.0, score + 0.25)
    if signals.products_mentioned:
        score = min(1.0, score + 0.1)

    signals.qualification_score = round(score, 2)
    signals.is_qualified = score >= 0.55 and (
        intent == VisitorIntent.READY_TO_ENROLL
        or len(signals.buying_signals) >= 2
        or bool(signals.email or signals.phone)
    )

    if signals.is_qualified:
        signals.sales_action = "follow_up_enrollment"
    elif intent == VisitorIntent.HESITANT:
        signals.sales_action = "nurture_address_objections"
    elif intent == VisitorIntent.PRICE_SENSITIVE:
        signals.sales_action = "share_pricing_options"
    else:
        signals.sales_action = "continue_discovery"

    return signals
