import pytest
from fastapi import HTTPException

from app.agent.graph import after_grade, after_grounding, after_route, finalize
from app.agent.tools import calculator
from app.auth import current_user
from app.config import Settings, get_settings
from app.ingest.pipeline import IngestError, check_source_url
from app.retrieval import Hit, reciprocal_rank_fusion


def hit(cid: int, source: str) -> Hit:
    return Hit(cid, "d", "Doc.pdf", 1, 1, None, f"chunk {cid}", 0.0, (source,))


def test_rrf_rewards_agreement():
    vector = [hit(1, "vector"), hit(2, "vector"), hit(3, "vector")]
    text = [hit(3, "fulltext"), hit(4, "fulltext")]
    fused = reciprocal_rank_fusion([vector, text], k=60)
    assert fused[0].chunk_id == 3  # in both lists
    assert fused[0].sources == ("fulltext", "vector")
    assert {h.chunk_id for h in fused} == {1, 2, 3, 4}


@pytest.mark.parametrize(
    "expr,expected",
    [("2 + 3 * 4", "14"), ("(1250 - 980) / 980 * 100", "27.55102"), ("2 ** 10", "1024")],
)
def test_calculator(expr, expected):
    assert expected in calculator.invoke({"expression": expr})


@pytest.mark.parametrize(
    "expr",
    ["__import__('os').system('id')", "open('/etc/passwd')", "[1] * 10", "9 ** 9 ** 9", "1/0"],
)
def test_calculator_refuses_unsafe(expr):
    assert calculator.invoke({"expression": expr}).startswith("error")


@pytest.mark.parametrize(
    "url",
    [
        "https://utfs.io/f/abc.pdf",
        "https://x7.ufs.sh/f/abc.pdf",
    ],
)
def test_source_url_allowed(url):
    check_source_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://utfs.io/f/abc.pdf",  # not https
        "https://169.254.169.254/latest/meta-data",
        "https://localhost/x.pdf",
        "https://utfs.io.evil.com/x.pdf",
        "https://evilutfs.io/x.pdf",
    ],
)
def test_source_url_blocked(url):
    with pytest.raises(IngestError):
        check_source_url(url)


def test_dev_auth_refused_in_production(monkeypatch):
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("AUTH_MODE", "dev")
    get_settings.cache_clear()
    try:
        with pytest.raises(RuntimeError):
            get_settings()
    finally:
        monkeypatch.setenv("ENV", "test")
        get_settings.cache_clear()


async def test_dev_auth_needs_header():
    assert (await current_user(None, "u1", None, None)).id == "u1"
    with pytest.raises(HTTPException) as exc:
        await current_user(None, None, None, None)
    assert exc.value.status_code == 401


async def test_kinde_mode_rejects_missing_and_bad_tokens(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "kinde")
    monkeypatch.setenv("KINDE_ISSUER_URL", "https://example.kinde.com")
    get_settings.cache_clear()
    try:
        for header in (None, "Basic abc", "Bearer not-a-jwt"):
            with pytest.raises(HTTPException) as exc:
                await current_user(
                    authorization=header,
                    x_dev_user="attacker",
                    x_service_token=None,
                    x_user_id=None,
                )
            assert exc.value.status_code == 401
    finally:
        monkeypatch.setenv("AUTH_MODE", "dev")
        get_settings.cache_clear()


def test_edges():
    assert after_route({"route": "direct"}) == "direct"
    assert after_route({"route": "tools"}) == "tool_agent"
    assert after_route({"route": "weird"}) == "retrieve"
    assert after_grade({"hits": [1]}) == "generate"
    assert after_grade({"hits": [], "rewrites": 0}) == "rewrite"
    assert after_grade({"hits": [], "rewrites": 1}) == "not_found"
    assert after_grounding({"strict": False}) == "finalize"
    assert (
        after_grounding({"strict": True, "grounding_attempts": 1, "route": "retrieve"})
        == "generate"
    )
    assert after_grounding({"strict": True, "grounding_attempts": 2}) == "finalize"


async def test_finalize_maps_labels_to_pages():
    hits = [
        {"document_id": "a", "document_name": "A.pdf", "page_start": 4, "page_end": 4},
        {"document_id": "a", "document_name": "A.pdf", "page_start": 7, "page_end": 8},
    ]
    out = await finalize({"answer": "Every 90 days [S1]. Logged [S2]. Bogus [S9].", "hits": hits})
    assert out["answer"] == "Every 90 days [p. 4]. Logged [pp. 7-8]. Bogus ."
    assert [s["label"] for s in out["sources"]] == ["S1", "S2"]


async def test_finalize_names_documents_when_several():
    hits = [
        {"document_id": "a", "document_name": "A.pdf", "page_start": 1, "page_end": 1},
        {"document_id": "b", "document_name": "B.pdf", "page_start": 2, "page_end": 2},
    ]
    out = await finalize({"answer": "X [S1] and Y [S2].", "hits": hits})
    assert out["answer"] == "X [A.pdf, p. 1] and Y [B.pdf, p. 2]."


def test_settings_defaults_are_safe(monkeypatch):
    monkeypatch.delenv("AUTH_MODE")
    s = Settings(_env_file=None)
    assert s.auth_mode == "kinde"
    assert "utfs.io" in s.allowed_source_hosts


async def test_service_token(monkeypatch):
    token = "s" * 40
    monkeypatch.setenv("AUTH_MODE", "kinde")
    monkeypatch.setenv("SERVICE_TOKEN", token)
    get_settings.cache_clear()
    try:
        user = await current_user(
            authorization=None, x_dev_user=None, x_service_token=token, x_user_id="kp_123"
        )
        assert user.id == "kp_123"
        for bad_token, uid, code in (
            ("wrong" * 8, "kp_123", 401),
            (token, None, 400),
            (token, "a b;drop", 400),
        ):
            with pytest.raises(HTTPException) as exc:
                await current_user(
                    authorization=None, x_dev_user=None, x_service_token=bad_token, x_user_id=uid
                )
            assert exc.value.status_code == code
    finally:
        monkeypatch.delenv("SERVICE_TOKEN")
        monkeypatch.setenv("AUTH_MODE", "dev")
        get_settings.cache_clear()


def test_short_service_token_refused(monkeypatch):
    monkeypatch.setenv("SERVICE_TOKEN", "short")
    get_settings.cache_clear()
    try:
        with pytest.raises(RuntimeError):
            get_settings()
    finally:
        monkeypatch.delenv("SERVICE_TOKEN")
        get_settings.cache_clear()
