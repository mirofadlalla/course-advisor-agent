"""Tests for sales agent prompts (phases 3, 5, 6, 13, 14)."""

from app.prompts import SYSTEM_PROMPT
from app.prompts.identity import IDENTITY
from app.prompts.safety import SAFETY
from app.prompts.sales import SALES


class TestSalesPrompts:
    def test_identity_is_sales_advisor(self):
        assert "Sales Advisor" in IDENTITY
        assert "Kayfa" in IDENTITY

    def test_sales_covers_product_tiers(self):
        for phrase in ("Free", "Individual paid courses", "tracks", "diplomas"):
            assert phrase.lower() in SALES.lower()

    def test_sales_covers_objections_and_policy(self):
        assert "objection" in SALES.lower() or "OBJECTION" in SALES
        assert "refund" in SALES.lower() or "policy" in SALES.lower()

    def test_safety_has_injection_and_role_guards(self):
        assert "PROMPT INJECTION" in SAFETY
        assert "sales-advisor role" in SAFETY.lower()

    def test_system_prompt_includes_sales_section(self):
        assert SALES.strip()[:40] in SYSTEM_PROMPT

    def test_behavior_dialect_guidance_in_system_prompt(self):
        from app.prompts.behavior import BEHAVIOR

        assert "DIALECT" in BEHAVIOR
        assert "Egyptian" in BEHAVIOR or "EG" in BEHAVIOR
