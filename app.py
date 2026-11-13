"""
Convertible bond pricer - Streamlit dashboard

    streamlit run app.py

Terms and market inputs in the sidebar; price, premiums and Greeks on top;
price profile, convertible-arbitrage P&L and a spot x credit scenario grid below.
"""

from dataclasses import replace

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from cb_pricer import (Convertible, Market, analytics, price, bond_floor,
                       hedged_pnl, implied_vol, load_underlying)

st.set_page_config(page_title="Convertible bond pricer", layout="wide")
st.title("Convertible bond pricer")
st.caption("Tsiveriotis-Fernandes binomial tree: equity part discounted at the risk-free "
           "rate, debt part at the risk-free rate plus the issuer's credit spread.")

# ------------------------------------------------------------------ inputs ---
sb = st.sidebar
sb.header("Underlying")
ticker = sb.text_input("Yahoo ticker (optional, e.g. AIR.PA)", "")
spot, vol, src = 100.0, 0.30, "explicit inputs"
if ticker:
    try:
        spot, vol, asof = load_underlying(ticker)
        src = f"{ticker}, close {asof}, 1-year realised vol"
    except Exception as exc:
        sb.warning(f"Market data unavailable ({exc}); explicit inputs used.")
spot = sb.number_input("Spot", value=round(spot, 2), min_value=0.01)
vol = sb.slider("Volatility", 0.05, 0.80, float(round(vol, 2)), 0.01)
div = sb.slider("Dividend yield", 0.0, 0.08, 0.015, 0.0025, format="%.4f")

sb.header("Rates and credit")
rate = sb.slider("Risk-free rate", 0.0, 0.08, 0.025, 0.0025, format="%.4f")
spread_bp = sb.slider("Issuer credit spread (bp)", 0, 1200, 250, 10)

sb.header("Bond terms")
maturity = sb.slider("Maturity (years)", 1.0, 10.0, 5.0, 0.5)
coupon = sb.slider("Coupon", 0.0, 0.08, 0.010, 0.0025, format="%.4f")
freq = sb.selectbox("Coupons per year", [1, 2, 4], index=1)
premium = sb.slider("Conversion premium at issue", 0.0, 1.0, 0.35, 0.05)
call_on = sb.checkbox("Issuer call", value=True)
call_start = sb.slider("Callable from (years)", 0.0, maturity, min(3.0, maturity), 0.5)
trigger = sb.slider("Soft-call trigger (parity, % of face; 0 = hard call)", 0.0, 1.6, 1.30, 0.05)
put_on = sb.checkbox("Investor put", value=False)
put_date = sb.slider("Put date (years)", 0.5, maturity, min(3.0, maturity), 0.5)
steps = sb.select_slider("Tree steps", [200, 400, 800], value=400)

cb = Convertible(maturity=maturity, coupon=coupon, freq=freq,
                 conv_price=round(spot * (1 + premium), 4),
                 call_start=call_start if call_on else None, call_trigger=trigger,
                 put_date=put_date if put_on else None)

# ------------------------------------------- terms read from a prospectus (RAG) ---
terms_src = "manual inputs in the sidebar"
with st.expander("Read the bond terms from a prospectus (PDF, Markdown or text)"):
    up = st.file_uploader("Terms and conditions", type=["pdf", "md", "txt"])
    if up is not None:
        import tempfile
        from pathlib import Path
        from prospectus_rag import ProspectusIndex, answer, extract_terms, to_convertible
        suffix = Path(up.name).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(up.getvalue())
        idx = ProspectusIndex(tmp.name)
        ex = extract_terms(idx)
        st.write(f"{len(idx.chunks)} passages, embeddings: {idx.embedder.backend}, "
                 f"extraction: {ex['method']}")
        st.table([{"term": k, "value": str(v), "passage": ex["sources"].get(k)}
                  for k, v in ex["terms"].items()])
        try:
            cb = to_convertible(ex["terms"])
            terms_src = f"extracted from {up.name}"
            st.success("The pricer now uses these terms. Set the spot in the sidebar to the share price (conversion price " f"{cb.conv_price:.2f}).")
        except KeyError as exc:
            st.warning(f"Term not found: {exc}. Manual terms kept.")
        q = st.text_input("Ask the document", "Is there dividend protection?")
        if q:
            a_ = answer(idx, q)
            st.write(a_["answer"])
            st.caption(f"{a_['mode']} | sources (passage, page, score): {a_['sources']}")

mk = Market(spot=spot, vol=vol, rate=rate, div=div, spread=spread_bp / 1e4)


@st.cache_data(show_spinner=False)
def run(cb, mk, steps):
    a = analytics(cb, mk, steps)
    spots = np.linspace(0.3, 2.2, 30) * mk.spot
    profile = [price(cb, replace(mk, spot=s), steps) for s in spots]
    pnl = hedged_pnl(cb, mk, np.linspace(-0.3, 0.3, 21), steps)
    moves = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    spreads = [100, 250, 400, 600, 900]
    grid = [[price(cb, replace(mk, spot=mk.spot * (1 + m), spread=s / 1e4), steps)
             for m in moves] for s in spreads]
    return a, spots, profile, pnl, moves, spreads, grid


a, spots, profile, pnl, moves, spreads, grid = run(cb, mk, steps)
st.caption(f"Market inputs: {src}. Bond terms: {terms_src}.")

# ------------------------------------------------------------------ screen ---
c = st.columns(6)
c[0].metric("Price", f"{a['price']:.2f}")
c[1].metric("Parity", f"{a['parity']:.2f}")
c[2].metric("Bond floor", f"{a['bond_floor']:.2f}")
c[3].metric("Conversion premium", f"{a['conversion_premium']:.1%}")
c[4].metric("Delta", f"{a['delta']:.1%}")
c[5].metric("Shares to short / bond", f"{a['delta_shares']:.4f}")
c = st.columns(6)
c[0].metric("Gamma (pts per 1%)", f"{a['gamma'] * 100:.2f}")
c[1].metric("Vega (per vol pt)", f"{a['vega']:.3f}")
c[2].metric("CS01 (per bp)", f"{a['cs01']:.4f}")
c[3].metric("Rho (per bp)", f"{a['rho']:.4f}")
c[4].metric("Equity part", f"{a['equity_part']:.2f}")
c[5].metric("Debt part", f"{a['debt_part']:.2f}")

# ------------------------------------------------------------------ charts ---
left, right = st.columns(2)
fig = go.Figure()
fig.add_trace(go.Scatter(x=spots, y=profile, name="Convertible", line=dict(width=3)))
fig.add_trace(go.Scatter(x=spots, y=cb.ratio * spots, name="Parity", line=dict(dash="dash")))
fig.add_trace(go.Scatter(x=spots, y=[bond_floor(cb, mk)] * len(spots), name="Bond floor",
                         line=dict(dash="dot")))
fig.add_vline(x=mk.spot, line_width=1, line_color="grey")
fig.update_layout(title="Price profile", xaxis_title="Share price",
                  yaxis_title="% of face", height=380, margin=dict(t=50, b=40))
left.plotly_chart(fig, width="stretch")

fig = go.Figure(go.Scatter(x=[m * 100 for m, _ in pnl], y=[p for _, p in pnl],
                           line=dict(width=3)))
fig.add_hline(y=0, line_width=1)
fig.update_layout(title="Convertible arbitrage: long bond, short delta",
                  xaxis_title="Instant share move (%)", yaxis_title="P&L per bond",
                  height=380, margin=dict(t=50, b=40))
right.plotly_chart(fig, width="stretch")

fig = go.Figure(go.Heatmap(z=grid, x=[f"{m:+.0%}" for m in moves],
                           y=[f"{s} bp" for s in spreads], colorscale="RdYlGn",
                           text=[[f"{v:.1f}" for v in row] for row in grid],
                           texttemplate="%{text}"))
fig.update_layout(title="Price under spot and credit scenarios", xaxis_title="Share move",
                  yaxis_title="Credit spread", height=360, margin=dict(t=50, b=40))
st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------- implied vol ---
st.subheader("Implied volatility from a market quote")
quote = st.number_input("Market price of the convertible", value=round(a["price"] - 1.0, 2))
try:
    iv = implied_vol(cb, mk, quote, steps=min(steps, 400))
    st.write(f"Implied volatility **{iv:.2%}** against {vol:.2%} used "
             f"({(iv - vol) * 100:+.1f} vol points).")
except ValueError as exc:
    st.write(str(exc))
