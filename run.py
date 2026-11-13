"""
Convertible bond report: price, analytics, scenarios and convertible arbitrage.

    python run.py                  illustrative convertible, explicit market inputs
    python run.py --ticker AIR.PA  spot and 1-year realised vol of a real underlying
                                   (Yahoo Finance); the bond terms stay illustrative

Writes out_summary.md, out_scenarios.csv and three charts.
"""

import argparse
from dataclasses import replace

import numpy as np

from cb_pricer import (Convertible, Market, analytics, price, bond_floor,
                       implied_vol, hedged_pnl, load_underlying)


def pct(x):
    return f"{x:.2%}"


def build(ticker=None):
    cb = Convertible()
    mk = Market()
    source = "Explicit inputs (illustrative convertible and market)"
    if ticker:
        try:
            spot, vol, asof = load_underlying(ticker)
            # keep the bond issued at a 35% premium to today's spot
            cb = replace(cb, conv_price=round(spot * 1.35, 2))
            mk = replace(mk, spot=spot, vol=vol)
            source = (f"{ticker}: last close {spot:.2f} and 1-year realised vol {vol:.2%} "
                      f"(Yahoo Finance, {asof}); bond terms illustrative")
        except Exception as exc:
            print(f"[!] Market data unavailable ({exc}); explicit inputs used.")
    return cb, mk, source


def report(cb, mk, source):
    a = analytics(cb, mk)
    L = ["# Convertible bond pricer - results", "", f"**Inputs:** {source}", "",
         "## Terms", "",
         "| Term | Value |", "|---|---|",
         f"| Face / maturity | {cb.face:.0f} / {cb.maturity:.0f} years |",
         f"| Coupon | {pct(cb.coupon)} p.a., {cb.freq} per year |",
         f"| Conversion price / ratio | {cb.conv_price:.2f} / {cb.ratio:.4f} shares |",
         f"| Issuer call | from year {cb.call_start} at {cb.call_price:.0f}, "
         f"soft call trigger {cb.call_trigger:.0%} of face |" if cb.call_start is not None
         else "| Issuer call | none |",
         f"| Investor put | {cb.put_date} at {cb.put_price:.0f} |" if cb.put_date
         else "| Investor put | none |",
         f"| Spot / vol | {mk.spot:.2f} / {pct(mk.vol)} |",
         f"| Rate / dividend / credit spread | {pct(mk.rate)} / {pct(mk.div)} / "
         f"{mk.spread * 1e4:.0f} bp |", "",
         "## Screen", "", "| Measure | Value |", "|---|---|",
         f"| Price | {a['price']:.2f} |",
         f"| Equity part / debt part (TF split) | {a['equity_part']:.2f} / {a['debt_part']:.2f} |",
         f"| Parity | {a['parity']:.2f} |",
         f"| Bond floor | {a['bond_floor']:.2f} |",
         f"| Conversion premium | {pct(a['conversion_premium'])} |",
         f"| Premium over bond floor | {pct(a['investment_premium'])} |",
         f"| Delta (share of parity) | {pct(a['delta'])} |",
         f"| Shares to short per bond | {a['delta_shares']:.4f} |",
         f"| Gamma (delta change per 1% spot) | {a['gamma'] * 100:.2f} pts |",
         f"| Vega (per vol point) | {a['vega']:.3f} |",
         f"| Rho (per bp) | {a['rho']:.4f} |",
         f"| CS01 (per bp of credit spread) | {a['cs01']:.4f} |", ""]

    # what each feature is worth
    no_call = price(replace(cb, call_start=None), mk)
    hard = price(replace(cb, call_trigger=0.0), mk)
    with_put = price(replace(cb, put_date=3.0), mk)
    L += ["## What each clause is worth", "", "| Structure | Price | vs base |", "|---|---|---|",
          f"| No issuer call | {no_call:.2f} | {no_call - a['price']:+.2f} |",
          f"| Base: soft call from year {cb.call_start} | {a['price']:.2f} | - |",
          f"| Hard call from year {cb.call_start} | {hard:.2f} | {hard - a['price']:+.2f} |",
          f"| Base + investor put at year 3 | {with_put:.2f} | {with_put - a['price']:+.2f} |", ""]

    # implied vol round trip at a market price 1 point below model
    mkt = a["price"] - 1.0
    iv = implied_vol(cb, mk, mkt)
    L += ["## Implied volatility", "",
          f"A bond quoted at {mkt:.2f}, one point under model, implies a volatility of "
          f"{pct(iv)} against {pct(mk.vol)} used: the market is pricing the option "
          f"{(mk.vol - iv) * 100:.1f} vol points cheaper (or the credit wider).", ""]
    return a, L


def scenarios(cb, mk):
    moves = [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30]
    spreads = [0.0100, 0.0250, 0.0400, 0.0600, 0.0900]
    grid = np.array([[price(cb, replace(mk, spot=mk.spot * (1 + m), spread=s), 500)
                      for m in moves] for s in spreads])
    return moves, spreads, grid


def charts(cb, mk, moves, spreads, grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spots = np.linspace(0.3, 2.2, 39) * mk.spot
    prices = [price(cb, replace(mk, spot=s), 500) for s in spots]
    floors = [bond_floor(cb, mk)] * len(spots)
    parity = cb.ratio * spots
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(spots, prices, color="#1F4E8C", lw=2, label="Convertible")
    ax.plot(spots, parity, "--", color="#C0504D", label="Parity (conversion value)")
    ax.plot(spots, floors, ":", color="black", label="Bond floor")
    ax.axvline(mk.spot, color="grey", lw=0.6)
    ax.set_xlabel("Share price")
    ax.set_ylabel("Price (% of face)")
    ax.set_title("A convertible: bond floor below, parity above")
    ax.legend()
    fig.tight_layout()
    fig.savefig("out_price_profile.png", dpi=150)
    plt.close(fig)

    pnl = hedged_pnl(cb, mk)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot([m * 100 for m, _ in pnl], [p for _, p in pnl], color="#1F4E8C", lw=2)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Instant move in the share price (%)")
    ax.set_ylabel("P&L per bond")
    ax.set_title("Convertible arbitrage: long the bond, short delta - long gamma")
    fig.tight_layout()
    fig.savefig("out_arbitrage_pnl.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    im = ax.imshow(grid, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(moves)), [f"{m:+.0%}" for m in moves])
    ax.set_yticks(range(len(spreads)), [f"{s * 1e4:.0f} bp" for s in spreads])
    for i in range(len(spreads)):
        for j in range(len(moves)):
            ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("Share price move")
    ax.set_ylabel("Credit spread")
    ax.set_title("Price under spot and credit scenarios")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig("out_scenarios.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=None)
    args = ap.parse_args()
    cb, mk, source = build(args.ticker)
    a, L = report(cb, mk, source)
    moves, spreads, grid = scenarios(cb, mk)
    L += ["## Scenarios (price, spot move x credit spread)", "",
          "| spread | " + " | ".join(f"{m:+.0%}" for m in moves) + " |",
          "|---|" + "---|" * len(moves)]
    for s, row in zip(spreads, grid):
        L.append(f"| {s * 1e4:.0f} bp | " + " | ".join(f"{v:.2f}" for v in row) + " |")
    with open("out_scenarios.csv", "w") as f:
        f.write("spread_bp," + ",".join(f"{m:+.0%}" for m in moves) + "\n")
        for s, row in zip(spreads, grid):
            f.write(f"{s * 1e4:.0f}," + ",".join(f"{v:.4f}" for v in row) + "\n")
    with open("out_summary.md", "w") as f:
        f.write("\n".join(L) + "\n")
    charts(cb, mk, moves, spreads, grid)
    print("\n".join(L))
    print("\nwritten: out_summary.md, out_scenarios.csv, out_price_profile.png, "
          "out_arbitrage_pnl.png, out_scenarios.png")


if __name__ == "__main__":
    main()
