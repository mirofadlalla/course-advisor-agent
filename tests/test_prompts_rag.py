"""Tests for Phase 2 RAG grounding prompts."""

from app.prompts import SYSTEM_PROMPT
from app.prompts.behavior import BEHAVIOR
from app.prompts.rag import RAG_PROMPT
from app.prompts.safety import SAFETY


class TestRagGroundingPrompts:
    def test_safety_lists_prohibited_hallucination_fields(self):
        for phrase in (
            "Prices",
            "refund",
            "Certificates",
            "Contact details",
            "Roadmaps",
        ):
            assert phrase.lower() in SAFETY.lower()

    def test_rag_requires_tool_before_factual_claims(self):
        assert "tool" in RAG_PROMPT.lower()
        assert "search_knowledge" in RAG_PROMPT
        assert "get_course_by_name" in RAG_PROMPT

    def test_rag_unknown_answer_includes_kayfa_contact_fallback(self):
        for contact in (
            "info@kayfa.io",
            "support@kayfa.io",
            "https://kayfa.io/contact-us/",
        ):
            assert contact in RAG_PROMPT

    def test_behavior_unknown_answer_directs_to_kayfa(self):
        assert "info@kayfa.io" in BEHAVIOR
        assert "support@kayfa.io" in BEHAVIOR
        assert "knowledge base" in BEHAVIOR.lower()

    def test_system_prompt_includes_grounding_sections(self):
        assert SAFETY.strip()[:40] in SYSTEM_PROMPT
        assert RAG_PROMPT.strip()[:40] in SYSTEM_PROMPT
        assert "info@kayfa.io" in SYSTEM_PROMPT
