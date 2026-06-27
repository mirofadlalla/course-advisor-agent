"""Tests for visitor intent detection."""

from app.sales.intent import VisitorIntent, build_intent_instructions, detect_intent


class TestIntentDetection:
    def test_browsing_english(self):
        assert detect_intent("What courses do you have?") == VisitorIntent.BROWSING

    def test_comparing(self):
        assert detect_intent("Compare Python vs Fullstack track") == VisitorIntent.COMPARING

    def test_price_sensitive(self):
        assert detect_intent("How much does the diploma cost?") == VisitorIntent.PRICE_SENSITIVE

    def test_hesitant(self):
        assert detect_intent("I'm not sure yet, maybe later") == VisitorIntent.HESITANT

    def test_ready_to_enroll(self):
        assert detect_intent("I want to enroll in the data science diploma") == (
            VisitorIntent.READY_TO_ENROLL
        )

    def test_history_influences_intent(self):
        history = ["How much is the Python course?"]
        assert detect_intent("Ok I want to register now", history) == (
            VisitorIntent.READY_TO_ENROLL
        )

    def test_build_intent_instructions_non_empty(self):
        for intent in VisitorIntent:
            text = build_intent_instructions(intent)
            assert intent.value.split("_")[0] in text.lower() or "Visitor intent" in text
