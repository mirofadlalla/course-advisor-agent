"""Tests for contextual lead detection."""

from app.sales.intent import VisitorIntent
from app.sales.lead_detector import detect_lead_signals


class TestLeadDetector:
    def test_price_question_adds_signal(self):
        signals = detect_lead_signals("How much is the Python course?")
        assert "asked_price" in signals.buying_signals
        assert not signals.is_qualified

    def test_qualified_lead_with_contact(self):
        signals = detect_lead_signals(
            "I want to enroll. My email is student@example.com. How do I pay?"
        )
        assert signals.is_qualified
        assert signals.email == "student@example.com"
        assert signals.qualification_score >= 0.55

    def test_products_extracted(self):
        signals = detect_lead_signals("Tell me about the Fullstack diploma")
        assert any("fullstack" in p.lower() for p in signals.products_mentioned)

    def test_arabic_dialect_egyptian(self):
        signals = detect_lead_signals("عايز اسجل في كورس Python")
        assert signals.language == "ar"
        assert signals.dialect == "egyptian"

    def test_objection_detected(self):
        signals = detect_lead_signals("It's too expensive for me")
        assert "price_objection" in signals.objections

    def test_history_context(self):
        history = ["What diplomas do you offer?"]
        signals = detect_lead_signals("What's the price?", history)
        assert "asked_price" in signals.buying_signals
