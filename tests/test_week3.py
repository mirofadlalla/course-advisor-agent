"""Week 3 feature tests — auth, persistence, cost, cancellation, admin."""

from __future__ import annotations

import asyncio

import pytest

from app.auth.jwt_handler import create_access_token, decode_token
from app.auth.password import hash_password, verify_password
from app.auth.service import AuthService, InvalidCredentialsError
from app.config import Settings
from app.cost.calculator import calculate_llm_cost, calculate_embedding_cost, merge_costs
from app.cost.pricing import get_pricing
from app.database.mongo import create_mongo_database
from app.repositories.user_repository import create_user_repository
from app.repositories.conversation_repository import create_conversation_repository
from app.repositories.message_repository import create_message_repository
from app.repositories.usage_log_repository import create_usage_log_repository
from app.repositories.trace_repository import create_trace_repository
from app.schemas.conversation import ConversationCreate
from app.schemas.trace import ResponseTrace, ToolCallTrace
from app.schemas.user import User, UserCreate, UserLogin, UserRole
from app.services.cancellation_manager import CancellationManager
from app.services.conversation_service import ConversationService
from app.services.usage_service import UsageService
from app.tracing.collector import TraceCollector


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        groq_api_key="test-key",
        jwt_secret_key="test-secret-key-for-jwt-signing-32chars",
        auth_disabled=True,
    )


@pytest.fixture
def mongo_db():
    return create_mongo_database("", "test_db")


# ── Authentication ────────────────────────────────────────────────────────────


def test_password_hashing():
    hashed = hash_password("secret123")
    assert verify_password("secret123", hashed)
    assert not verify_password("wrong", hashed)


def test_jwt_roundtrip(test_settings: Settings):
    token = create_access_token(
        test_settings,
        user_id="u1",
        email="a@test.com",
        role=UserRole.USER,
    )
    payload = decode_token(test_settings, token, expected_type="access")
    assert payload.sub == "u1"
    assert payload.email == "a@test.com"


@pytest.mark.asyncio
async def test_register_and_login(test_settings: Settings, mongo_db):
    user_repo = create_user_repository(mongo_db)
    auth = AuthService(user_repo, test_settings)
    user, tokens = await auth.register(
        UserCreate(full_name="Admin", email="admin@test.com", password="pass", role=UserRole.ADMIN)
    )
    assert user.role == UserRole.ADMIN
    assert tokens.access_token
    logged_in, login_tokens = await auth.login(
        UserLogin(email="admin@test.com", password="pass")
    )
    assert logged_in.email == "admin@test.com"
    assert login_tokens.refresh_token


@pytest.mark.asyncio
async def test_invalid_login(test_settings: Settings, mongo_db):
    user_repo = create_user_repository(mongo_db)
    auth = AuthService(user_repo, test_settings)
    await auth.register(UserCreate(full_name="U", email="u@test.com", password="pass"))
    with pytest.raises(InvalidCredentialsError):
        await auth.login(UserLogin(email="u@test.com", password="bad"))


# ── Role protection ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_admin_role_check(test_settings: Settings, mongo_db):
    """Non-admin JWT must be rejected by require_admin logic."""
    from app.auth.jwt_handler import create_access_token, decode_token
    from app.schemas.user import UserRole

    user_token = create_access_token(
        test_settings, user_id="u1", email="u@test.com", role=UserRole.USER
    )
    payload = decode_token(test_settings, user_token, expected_type="access")
    assert payload.role != UserRole.ADMIN.value

    admin_token = create_access_token(
        test_settings, user_id="a1", email="a@test.com", role=UserRole.ADMIN
    )
    admin_payload = decode_token(test_settings, admin_token, expected_type="access")
    assert admin_payload.role == UserRole.ADMIN.value


# ── Conversation persistence ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_conversation_and_messages(mongo_db):
    conv_repo = create_conversation_repository(mongo_db)
    msg_repo = create_message_repository(mongo_db)
    service = ConversationService(conv_repo, msg_repo)
    conv = await service.create_conversation("user-1", ConversationCreate(title="Test"))
    await service.save_user_message(conv.conversation_id, "Hello")
    await service.save_assistant_message(conv.conversation_id, "Hi there", tokens=10, cost=0.001)
    history = await service.get_history(conv.conversation_id)
    assert len(history) == 2
    assert history[0].content == "Hello"
    assert history[1].tokens == 10


# ── Usage logging ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_usage_logging(mongo_db):
    repo = create_usage_log_repository(mongo_db)
    service = UsageService(repo)
    log = await service.log_model_call(
        user_id="u1",
        conversation_id="c1",
        message_id="m1",
        model_key="groq:llama-3.3-70b-versatile",
        prompt_tokens=100,
        completion_tokens=50,
        embedding_tokens=20,
        latency_ms=123.4,
    )
    assert log.total_tokens == 170
    assert log.provider == "Groq"
    assert log.total_cost >= 0


# ── Trace logging ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_trace_collector_and_repository(mongo_db):
    repo = create_trace_repository(mongo_db)
    collector = TraceCollector(
        user_id="u1",
        conversation_id="c1",
        message_id="m1",
        user_prompt="What courses?",
    )
    collector.set_intent("browsing", "Be helpful")
    collector.add_tool_call("search_knowledge", {"query": "courses"}, [{"text": "chunk"}])
    collector.set_response_metrics(
        response="Here are courses",
        latency_ms=200,
        prompt_tokens=50,
        completion_tokens=30,
        embedding_tokens=10,
        prompt_cost=0.0001,
        completion_cost=0.0002,
        embedding_cost=0.0,
        total_cost=0.0003,
        provider="Groq",
        model="llama-3.3-70b-versatile",
    )
    trace = collector.build()
    saved = await repo.insert(trace)
    loaded = await repo.get(saved.id)
    assert loaded is not None
    assert len(loaded.to_replay_steps()) > 5


# ── Cost calculation ──────────────────────────────────────────────────────────


def test_cost_calculation():
    pricing = get_pricing("groq:llama-3.3-70b-versatile")
    assert pricing.provider == "Groq"
    llm = calculate_llm_cost("groq:llama-3.3-70b-versatile", 1_000_000, 1_000_000)
    assert llm.prompt_cost == pytest.approx(0.59, rel=1e-3)
    assert llm.completion_cost == pytest.approx(0.79, rel=1e-3)
    emb = calculate_embedding_cost("BAAI/bge-m3", 1000)
    assert emb.total_cost == 0.0
    merged = merge_costs(llm, emb)
    assert merged.total_cost == pytest.approx(1.38, rel=1e-3)


# ── Cancellation ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancellation_manager():
    mgr = CancellationManager()
    results: list[str] = []

    async def long_task(name: str) -> None:
        try:
            await asyncio.sleep(10)
            results.append(name)
        except asyncio.CancelledError:
            results.append(f"{name}-cancelled")
            raise

    t1 = asyncio.create_task(long_task("first"))
    await mgr.register("conv-1", t1)
    await asyncio.sleep(0.05)
    t2 = asyncio.create_task(long_task("second"))
    await mgr.register("conv-1", t2)
    await asyncio.sleep(0.1)
    assert "first-cancelled" in results or t1.cancelled()
    t2.cancel()
    try:
        await t2
    except asyncio.CancelledError:
        pass
    await mgr.unregister("conv-1", t2)


# ── Trace replay steps ────────────────────────────────────────────────────────


def test_trace_replay_steps():
    trace = ResponseTrace(
        user_id="u1",
        conversation_id="c1",
        message_id="m1",
        user_prompt="Hello",
        intent="browsing",
        tool_calls=[ToolCallTrace(tool_name="search_knowledge", arguments={"q": "x"}, result=[])],
        llm_response="Answer",
    )
    steps = [s["step"] for s in trace.to_replay_steps()]
    assert "User Prompt" in steps
    assert "Assistant Reply" in steps
