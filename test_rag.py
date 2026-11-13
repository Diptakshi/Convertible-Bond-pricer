import json
from datetime import date
from pathlib import Path

import pytest

import prospectus_rag as rag

DOC = Path(__file__).parent.parent / "docs" / "sample_terms_illustrative.md"


@pytest.fixture(scope="module")
def index():
    return rag.ProspectusIndex(DOC, backend="lsa")


def test_chunks_overlap():
    chunks = rag.chunk([" ".join(f"w{i}" for i in range(200))])
    a, b = chunks[0].text.split(), chunks[1].text.split()
    assert a[-25:] == b[:25]                         # 25-word overlap between windows


def test_retrieval_finds_the_right_clause(index):
    for q, word in [("Is there dividend protection?", "Dividend Protection"),
                    ("What happens on a change of control?", "Change of Control"),
                    ("What is the conversion price?", "Conversion Price")]:
        assert any(word in c.text for c, _ in index.search(q, 2))


def test_extracts_every_term(index):
    t = rag.extract_terms(index, use_llm=False)["terms"]
    assert t["issue_date"] == date(2026, 1, 15)
    assert t["maturity_date"] == date(2031, 1, 15)
    assert t["coupon"] == pytest.approx(0.0125) and t["freq"] == 2
    assert t["conv_price"] == pytest.approx(54.0)
    assert t["call_date"] == date(2029, 2, 5) and t["call_trigger"] == pytest.approx(1.3)
    assert t["put_date"] == date(2029, 1, 15)


def test_terms_feed_the_pricer(index):
    cb = rag.to_convertible(rag.extract_terms(index, use_llm=False)["terms"])
    assert cb.maturity == pytest.approx(5.0, abs=0.01)
    assert cb.call_start == pytest.approx(3.06, abs=0.01)
    assert cb.put_date == pytest.approx(3.0, abs=0.01)
    assert cb.ratio == pytest.approx(100 / 54)


def test_answer_without_llm_is_extractive_and_cited(index, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    a = rag.answer(index, "When can the issuer call the bonds?")
    assert a["mode"].startswith("extractive") and len(a["sources"]) == 3


def test_llm_extraction_parses_json(index, monkeypatch):
    reply = {"issue_date": "2026-01-15", "maturity_date": "2031-01-15", "call_date": "2029-02-05",
             "put_date": None, "coupon": 0.0125, "freq": 2, "conv_price": 54.0,
             "call_trigger": 1.3, "sources": {"conv_price": 4}}
    monkeypatch.setattr(rag, "_llm", lambda prompt, max_tokens=600: "Here:\n" + json.dumps(reply))
    out = rag.extract_terms(index, use_llm=True)
    assert out["terms"]["call_date"] == date(2029, 2, 5)
    assert "put_date" not in out["terms"]
    assert out["sources"]["conv_price"] == 4
