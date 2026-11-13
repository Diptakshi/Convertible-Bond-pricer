"""
Prospectus reader for convertible bonds: retrieval-augmented generation (RAG)

Before pricing a new convertible, a trader reads its terms and conditions: conversion
price, coupon, maturity, call and put clauses. This module does that step and hands the
terms to the pricer.

    1. Load      PDF, Markdown or text terms and conditions
    2. Chunk     overlapping passages, each tagged with its page
    3. Embed     sentence-transformers if installed (multilingual MiniLM),
                 otherwise TF-IDF + truncated SVD (LSA) embeddings, fully offline
    4. Retrieve  cosine similarity between the question and every passage
    5. Answer    an LLM (Claude, if ANTHROPIC_API_KEY is set) answers from the retrieved
                 passages only and cites them; without a key, the best passage is returned
    6. Extract   one retrieval per term, then the LLM (JSON) or regular expressions
                 read the value; every term keeps the passage it came from
    7. Price     the extracted terms go straight into cb_pricer.Convertible

    python prospectus_rag.py docs/sample_terms_illustrative.md --spot 40 --vol 0.30
    python prospectus_rag.py my_terms.pdf --ask "Is there dividend protection?"

Diptakshi Sen - personal project, 2026
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

MONTHS = {m: i + 1 for i, m in enumerate(
    "january february march april may june july august september october november december".split())}
MONTHS.update({m: i + 1 for i, m in enumerate(
    "janvier février mars avril mai juin juillet août septembre octobre novembre décembre".split())})


# ============================================================ 1-2. LOAD AND CHUNK ===
@dataclass
class Chunk:
    id: int
    page: int
    text: str


def load_pages(path: str | Path) -> list[str]:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return [p.extract_text() or "" for p in PdfReader(str(path)).pages]
    return [path.read_text(encoding="utf-8")]


def chunk(pages: list[str], size: int = 70, overlap: int = 25) -> list[Chunk]:
    """Overlapping windows of words, so a clause heading stays with its text and a clause
    cut at a window edge appears whole in the next window."""
    out = []
    for p, text in enumerate(pages, start=1):
        words = re.sub(r"\s+", " ", text).split()
        start = 0
        while start < len(words):
            out.append(Chunk(len(out), p, " ".join(words[start:start + size])))
            if start + size >= len(words):
                break
            start += size - overlap
    return out


# ================================================================== 3. EMBED ===
class Embedder:
    """Sentence embeddings, with an offline fallback.

    backend "st"  : sentence-transformers, paraphrase-multilingual-MiniLM-L12-v2
                    (handles French and English terms and conditions)
    backend "lsa" : TF-IDF on words and character n-grams, reduced by truncated SVD;
                    no download, used when sentence-transformers is not available
    """

    def __init__(self, backend: str = "auto"):
        self.backend = backend
        self.model = None
        if backend in ("auto", "st"):
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
                self.backend = "st"
            except Exception:
                if backend == "st":
                    raise
                self.backend = "lsa"

    def fit(self, texts: list[str]):
        if self.backend == "lsa":
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.pipeline import make_pipeline, make_union
            vec = make_union(
                TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2)),
                TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(3, 5)))
            n = max(2, min(64, len(texts) - 1))
            self.lsa = make_pipeline(vec, TruncatedSVD(n_components=n, random_state=0))
            self.lsa.fit(texts)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.backend == "st":
            v = self.model.encode(texts, normalize_embeddings=True)
        else:
            v = self.lsa.transform(texts)
        v = np.asarray(v, float)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


# =============================================================== 4. RETRIEVE ===
class ProspectusIndex:
    def __init__(self, path: str | Path, backend: str = "auto"):
        self.path = Path(path)
        self.chunks = chunk(load_pages(path))
        if not self.chunks:
            raise ValueError(f"no text found in {path}")
        texts = [c.text for c in self.chunks]
        self.embedder = Embedder(backend).fit(texts)
        self.vectors = self.embedder.encode(texts)

    def search(self, question: str, k: int = 3) -> list[tuple[Chunk, float]]:
        q = self.embedder.encode([question])[0]
        scores = self.vectors @ q
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


# ================================================================= 5. ANSWER ===
def llm_available() -> bool:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


def _llm(prompt: str, max_tokens: int = 600) -> str:
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=os.environ.get("RAG_MODEL", "claude-sonnet-4-5"), max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text


def _context(hits) -> str:
    return "\n\n".join(f"[passage {c.id}, page {c.page}]\n{c.text}" for c, _ in hits)


def answer(index: ProspectusIndex, question: str, k: int = 3) -> dict:
    hits = index.search(question, k)
    if llm_available():
        prompt = ("You read convertible bond terms and conditions for a trading desk. Answer "
                  "the question using only the passages below, and cite the passage numbers "
                  "you rely on. If the passages do not contain the answer, say so.\n\n"
                  f"{_context(hits)}\n\nQuestion: {question}")
        text, mode = _llm(prompt), "LLM answer grounded in the retrieved passages"
    else:
        text, mode = hits[0][0].text, "extractive: best-matching passage (no LLM key set)"
    return {"question": question, "answer": text, "mode": mode,
            "sources": [(c.id, c.page, round(s, 3)) for c, s in hits]}


# ================================================================ 6. EXTRACT ===
QUERIES = {
    "issue_date": "issue date of the bonds, issued at 100 per cent",
    "maturity_date": "redemption at maturity date, bonds redeemed at principal amount",
    "coupon": "interest rate per cent per annum payable semi-annually coupon",
    "conv_price": "initial conversion price per share conversion ratio",
    "call": "redemption at the option of the issuer soft call per cent of the conversion price",
    "put": "redemption at the option of bondholders put date",
}

_NUM = r"(\d+(?:[.,]\d+)?)"
_DATE = r"(\d{1,2})\s+([A-Za-zéû]+)\s+(\d{4})"


def _num(s: str) -> float:
    return float(s.replace(",", "."))


def _dates(text: str) -> list[date]:
    out = []
    for d, m, y in re.findall(_DATE, text):
        if m.lower() in MONTHS:
            out.append(date(int(y), MONTHS[m.lower()], int(d)))
    return out


def _regex_terms(passages: dict[str, str]) -> dict:
    t = {}
    m = re.search(_DATE + r"[^.]{0,20}issue date", passages["issue_date"], re.I) \
        or re.search(r"date d.émission[^0-9]{0,20}" + _DATE, passages["issue_date"], re.I)
    if m:
        t["issue_date"] = _dates(m.group(0))[0]
    m = re.search(r"redeemed .*?on " + _DATE + r"|(?:maturity|échéance)[^.]*?" + _DATE,
                  passages["maturity_date"], re.I)
    if m:
        t["maturity_date"] = _dates(m.group(0))[0]
    m = re.search(_NUM + r"\s*(?:per cent\.?|%)\s*(?:per annum|par an|l.an)",
                  passages["coupon"], re.I)
    if m:
        t["coupon"] = _num(m.group(1)) / 100
    if re.search(r"semi-annual|semestriel", passages["coupon"], re.I):
        t["freq"] = 2
    elif re.search(r"quarterly|trimestriel", passages["coupon"], re.I):
        t["freq"] = 4
    elif "coupon" in t:
        t["freq"] = 1
    m = re.search(r"conversion price (?:is|of)\s*(?:EUR|€|USD|\$)?\s*" + _NUM, passages["conv_price"], re.I) \
        or re.search(r"prix de conversion[^0-9]*" + _NUM, passages["conv_price"], re.I)
    if m:
        t["conv_price"] = _num(m.group(1))
    call = passages["call"]
    m = re.search(r"(?:on or after|à compter du|from)\s+" + _DATE, call, re.I)
    if m:
        t["call_date"] = _dates(m.group(0))[0]
    m = re.search(r"exceeds?\s+" + _NUM + r"\s*(?:per cent\.?|%)", call, re.I) \
        or re.search(r"excède\s+" + _NUM + r"\s*%", call, re.I)
    if m:
        t["call_trigger"] = _num(m.group(1)) / 100
    elif "call_date" in t:
        t["call_trigger"] = 0.0                          # hard call
    put = passages["put"]
    m = re.search(r"(?:option of (?:the )?bondholders|au gré des porteurs)[^.]*?" + _DATE, put, re.I)
    if m:
        t["put_date"] = _dates(m.group(0))[-1]
    return t


def _llm_terms(index: ProspectusIndex) -> dict:
    hits = {}
    for q in QUERIES.values():
        for c, s in index.search(q, 2):
            hits[c.id] = (c, s)
    prompt = (
        "Extract the terms of this convertible bond from the passages. Reply with JSON only, "
        "keys: issue_date, maturity_date, call_date, put_date (YYYY-MM-DD or null), coupon "
        "(decimal, e.g. 0.0125), freq (coupons per year), conv_price (number), call_trigger "
        "(soft-call threshold as a multiple of the conversion price, e.g. 1.3; 0 for a hard "
        "call; null if no call), and sources (object: key -> passage number).\n\n"
        + _context(list(hits.values())))
    raw = _llm(prompt, 800)
    data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    out = {k: v for k, v in data.items() if v is not None and k != "sources"}
    for k in ("issue_date", "maturity_date", "call_date", "put_date"):
        if k in out:
            out[k] = date.fromisoformat(out[k])
    out["_sources"] = data.get("sources", {})
    return out


def extract_terms(index: ProspectusIndex, use_llm: bool | None = None) -> dict:
    """Terms of the convertible, each with the passage it was read from."""
    use_llm = llm_available() if use_llm is None else use_llm
    # context expansion: the two best passages per term plus their neighbours, in document
    # order, so a clause cut by a window edge is read whole
    def expand(q):
        ids = set()
        for c, _ in index.search(q, 2):
            ids |= {c.id - 1, c.id, c.id + 1}
        return [index.chunks[i] for i in sorted(ids) if 0 <= i < len(index.chunks)]
    top2 = {k: expand(q) for k, q in QUERIES.items()}
    best = {k: index.search(q, 1)[0][0] for k, q in QUERIES.items()}
    if use_llm:
        terms = _llm_terms(index)
        sources = terms.pop("_sources", {})
        method = "LLM extraction on retrieved passages"
    else:
        terms = _regex_terms({k: " ".join(c.text for c in cs) for k, cs in top2.items()})
        sources = {}
        method = "regular expressions on retrieved passages"
    src_key = {"issue_date": "issue_date", "maturity_date": "maturity_date", "coupon": "coupon",
               "freq": "coupon", "conv_price": "conv_price", "call_date": "call",
               "call_trigger": "call", "put_date": "put"}
    cites = {k: sources.get(k, best[src_key[k]].id) for k in terms if k in src_key}
    return {"terms": terms, "sources": cites, "method": method,
            "embeddings": index.embedder.backend}


def to_convertible(terms: dict, asof: date | None = None):
    """Map extracted terms to the pricer, in years from the valuation date."""
    from cb_pricer import Convertible
    t = terms
    asof = asof or t["issue_date"]
    yrs = lambda d: max((d - asof).days / 365.25, 0.0)
    return Convertible(
        face=100.0,
        maturity=round(yrs(t["maturity_date"]), 4),
        coupon=t.get("coupon", 0.0),
        freq=int(t.get("freq", 1)),
        conv_price=t["conv_price"],
        call_start=round(yrs(t["call_date"]), 4) if "call_date" in t else None,
        call_trigger=t.get("call_trigger", 0.0),
        put_date=round(yrs(t["put_date"]), 4) if "put_date" in t else None,
    )


# ================================================================ 7. PRICE ===
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("document")
    ap.add_argument("--ask", action="append", default=[])
    ap.add_argument("--spot", type=float, help="share price; default: conversion price / 1.35")
    ap.add_argument("--vol", type=float, default=0.30)
    ap.add_argument("--rate", type=float, default=0.025)
    ap.add_argument("--div", type=float, default=0.015)
    ap.add_argument("--spread", type=float, default=0.025)
    ap.add_argument("--backend", default="auto", choices=["auto", "st", "lsa"])
    args = ap.parse_args()

    idx = ProspectusIndex(args.document, args.backend)
    print(f"{args.document}: {len(idx.chunks)} passages, embeddings = {idx.embedder.backend}, "
          f"LLM = {'on' if llm_available() else 'off (set ANTHROPIC_API_KEY)'}\n")
    for q in args.ask:
        a = answer(idx, q)
        print(f"Q: {q}\nA ({a['mode']}): {a['answer']}\nSources (passage, page, score): "
              f"{a['sources']}\n")

    ex = extract_terms(idx)
    print(f"Extracted terms ({ex['method']}):")
    for k, v in ex["terms"].items():
        print(f"  {k:<14} {v!s:<12} <- passage {ex['sources'].get(k)}")
    from dataclasses import replace
    from cb_pricer import Market, analytics
    cb = to_convertible(ex["terms"])
    mk = Market(spot=args.spot or cb.conv_price / 1.35, vol=args.vol, rate=args.rate,
                div=args.div, spread=args.spread)
    a = analytics(cb, mk)
    print(f"\nPricer input: {cb}\nMarket: {mk}  (explicit inputs)")
    print(f"Price {a['price']:.2f} | parity {a['parity']:.2f} | bond floor {a['bond_floor']:.2f} | "
          f"premium {a['conversion_premium']:.1%} | delta {a['delta']:.1%} | "
          f"CS01 {a['cs01']:.4f}")


if __name__ == "__main__":
    main()
