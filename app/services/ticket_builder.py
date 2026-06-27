"""Build Arabic CRM ticket summaries with English product names preserved."""

from __future__ import annotations

from app.sales.lead_detector import LeadSignals
from app.sales.intent import VisitorIntent
from app.schemas.crm_ticket import CrmTicket

_INTENT_AR: dict[VisitorIntent, str] = {
    VisitorIntent.BROWSING: "يتصفح الخيارات",
    VisitorIntent.COMPARING: "يقارن بين البرامج",
    VisitorIntent.HESITANT: "متردد",
    VisitorIntent.PRICE_SENSITIVE: "حساس للسعر",
    VisitorIntent.READY_TO_ENROLL: "جاهز للتسجيل",
}

_SIGNAL_AR: dict[str, str] = {
    "asked_price": "سأل عن السعر",
    "asked_payment": "سأل عن الدفع",
    "asked_enrollment": "سأل عن التسجيل",
    "asked_start_date": "سأل عن موعد البداية",
    "asked_contact": "طلب التواصل",
    "ready_to_buy": "أبدى رغبة في الشراء",
}

_OBJECTION_AR: dict[str, str] = {
    "price_objection": "اعتراض على السعر",
    "time_objection": "اعتراض بسبب الوقت",
    "trust_objection": "تردد / ثقة",
}

_ACTION_AR: dict[str, str] = {
    "follow_up_enrollment": "متابعة للتسجيل",
    "nurture_address_objections": "معالجة الاعتراضات",
    "share_pricing_options": "مشاركة خيارات التسعير",
    "continue_discovery": "متابعة استكشاف الاحتياج",
}


def _join_ar(items: list[str], mapping: dict[str, str]) -> str:
    translated = [mapping.get(item, item) for item in items if item]
    return "، ".join(translated) if translated else "لا يوجد"


def build_arabic_summary(ticket: CrmTicket) -> str:
    """Generate a natural Arabic summary; keep product names in English."""
    intent_label = _INTENT_AR.get(
        VisitorIntent(ticket.visitor_intent) if ticket.visitor_intent else VisitorIntent.BROWSING,
        ticket.visitor_intent or "غير محدد",
    )
    products = "، ".join(ticket.products_interested) if ticket.products_interested else "غير محددة"
    signals = _join_ar(ticket.buying_signals, _SIGNAL_AR)
    objections = _join_ar(ticket.objections, _OBJECTION_AR)
    action = _ACTION_AR.get(ticket.sales_action, ticket.sales_action or "متابعة")

    contact_bits = []
    if ticket.name:
        contact_bits.append(f"الاسم: {ticket.name}")
    if ticket.phone:
        contact_bits.append(f"الهاتف: {ticket.phone}")
    if ticket.whatsapp:
        contact_bits.append(f"واتساب: {ticket.whatsapp}")
    if ticket.email:
        contact_bits.append(f"البريد: {ticket.email}")
    if ticket.city:
        contact_bits.append(f"المدينة: {ticket.city}")
    contact_line = " | ".join(contact_bits) if contact_bits else "لم يُذكر بعد"

    goal = ticket.goal or "غير محدد"
    level = ticket.level or "غير محدد"

    return (
        f"عميل محتمل من موقع Kayfa — {intent_label}.\n"
        f"المنتجات المهتم بها: {products}.\n"
        f"الهدف: {goal} | المستوى: {level}.\n"
        f"إشارات الشراء: {signals}.\n"
        f"الاعتراضات: {objections}.\n"
        f"بيانات التواصل: {contact_line}.\n"
        f"الإجراء المقترح: {action}."
    )


def build_ticket_from_signals(
    signals: LeadSignals,
    *,
    session_id: str,
    visitor_intent: VisitorIntent,
    conversation_excerpt: str,
) -> CrmTicket:
    """Map lead signals to a CRM ticket with Arabic summary."""
    ticket = CrmTicket(
        session_id=session_id,
        name=signals.name,
        phone=signals.phone,
        whatsapp=signals.whatsapp or signals.phone,
        email=signals.email,
        city=signals.city,
        language=signals.language,
        dialect=signals.dialect,
        contact_preference=signals.contact_preference,
        products_interested=signals.products_mentioned,
        goal=signals.goal,
        level=signals.level,
        buying_signals=signals.buying_signals,
        objections=signals.objections,
        visitor_intent=visitor_intent.value,
        sales_action=signals.sales_action,
        conversation_excerpt=conversation_excerpt[:2000],
    )
    ticket.arabic_summary = build_arabic_summary(ticket)
    return ticket
