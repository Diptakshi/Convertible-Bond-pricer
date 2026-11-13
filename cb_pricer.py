"""
Convertible bond pricer - Tsiveriotis-Fernandes binomial tree

A convertible is a bond plus the holder's right to swap it for shares. The two
pieces carry different risk: cash the issuer must repay is exposed to its
credit, shares delivered on conversion are not. Tsiveriotis and Fernandes (1998)
split the value at every node into an equity part, discounted at the risk-free
rate, and a debt part, discounted at the risk-free rate plus the issuer's credit
spread.

Features
    coupons (any frequency), dividend yield, credit spread
    issuer call: hard call, or soft call with a parity trigger
    investor put on a single date
    analytics: parity, bond floor, conversion and investment premium,
               delta, gamma, vega, rho, CS01, implied volatility

Diptakshi Sen - personal project, 2026
"""

from dataclasses import dataclass, replace
from math import exp, sqrt, log, erf

import numpy as np


@dataclass(frozen=True)
class Convertible:
    face: float = 100.0
    maturity: float = 5.0            # years
    coupon: float = 0.010            # annual coupon rate
    freq: int = 2                    # coupons per year
    conv_price: float = 135.0        # face / conversion ratio
    call_start: float | None = 3.0   # issuer call allowed from this date (None = no call)
    call_price: float = 100.0        # paid by the issuer on call, % of face
    call_trigger: float = 1.30       # soft call: parity must exceed trigger x face (0 = hard call)
    put_date: float | None = None    # investor put date (None = no put)
    put_price: float = 100.0

    @property
    def ratio(self):
        """Conversion ratio: shares received per bond."""
        return self.face / self.conv_price


@dataclass(frozen=True)
class Market:
    spot: float = 100.0
    vol: float = 0.30
    rate: float = 0.025
    div: float = 0.015
    spread: float = 0.025            # issuer credit spread over the risk-free rate


def _tree(cb: Convertible, mk: Market, steps: int):
    T, dt = cb.maturity, cb.maturity / steps
    u = exp(mk.vol * sqrt(dt))
    d = 1 / u
    p = (exp((mk.rate - mk.div) * dt) - d) / (u - d)
    if not 0 < p < 1:
        raise ValueError("tree probability outside (0, 1): use more steps")
    disc_e = exp(-mk.rate * dt)
    disc_d = exp(-(mk.rate + mk.spread) * dt)

    cpn = cb.face * cb.coupon / cb.freq
    n_cpn = int(round(cb.maturity * cb.freq))
    cpn_steps = {int(round((k + 1) / cb.freq / dt)) for k in range(n_cpn)}
    put_step = None if cb.put_date is None else int(round(cb.put_date / dt))

    j = np.arange(steps + 1)
    S = mk.spot * u ** (steps - 2 * j)
    conv = cb.ratio * S
    redeem = cb.face + (cpn if steps in cpn_steps else 0.0)
    convert = conv >= redeem
    E = np.where(convert, conv, 0.0)
    D = np.where(convert, 0.0, redeem)

    greeks = {}
    for n in range(steps - 1, -1, -1):
        t = n * dt
        S = mk.spot * u ** (n - 2 * np.arange(n + 1))
        E = disc_e * (p * E[:-1] + (1 - p) * E[1:])
        D = disc_d * (p * D[:-1] + (1 - p) * D[1:])
        if n in cpn_steps and n > 0:
            D = D + cpn
        V = E + D
        conv = cb.ratio * S

        # issuer call: the issuer calls when holding is worth more than the call price
        if cb.call_start is not None and t >= cb.call_start - 1e-12:
            allowed = conv >= cb.call_trigger * cb.face
            called = allowed & (V > cb.call_price)
            forced = called & (conv >= cb.call_price)      # holder converts instead
            redeemed = called & ~forced
            E = np.where(forced, conv, np.where(redeemed, 0.0, E))
            D = np.where(forced, 0.0, np.where(redeemed, cb.call_price, D))
            V = E + D

        # investor put
        if put_step is not None and n == put_step:
            put = V < cb.put_price
            E = np.where(put, 0.0, E)
            D = np.where(put, cb.put_price, D)
            V = E + D

        # voluntary conversion
        cv = conv > V
        E = np.where(cv, conv, E)
        D = np.where(cv, 0.0, D)
        if n in (1, 2):
            greeks[n] = (S.copy(), (E + D).copy())
    # delta and gamma read off the tree itself, so they do not inherit the
    # node-shift noise of bumping the spot
    (S1, V1), (S2, V2) = greeks[1], greeks[2]
    delta = (V1[0] - V1[1]) / (S1[0] - S1[1])
    gamma = ((V2[0] - V2[1]) / (S2[0] - S2[1]) - (V2[1] - V2[2]) / (S2[1] - S2[2])) \
        / (0.5 * (S2[0] - S2[2]))
    return float(E[0] + D[0]), float(E[0]), float(D[0]), float(delta), float(gamma)


def price(cb: Convertible, mk: Market, steps: int = 800, split=False):
    """Convertible price, averaged over steps and steps+1 to damp the
    odd/even oscillation of binomial trees."""
    a, b = _tree(cb, mk, steps), _tree(cb, mk, steps + 1)
    v, e, d, delta, gamma = [(x + y) / 2 for x, y in zip(a, b)]
    return (v, e, d, delta, gamma) if split else v


def bond_floor(cb: Convertible, mk: Market):
    """Straight bond value: coupons and redemption at the risky rate, no option."""
    y = mk.rate + mk.spread
    n = int(round(cb.maturity * cb.freq))
    cpn = cb.face * cb.coupon / cb.freq
    times = [(k + 1) / cb.freq for k in range(n)]
    return sum(cpn * exp(-y * t) for t in times) + cb.face * exp(-y * cb.maturity)


def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def bs_call(S, K, T, r, q, sigma):
    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * exp(-q * T) * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)


def analytics(cb: Convertible, mk: Market, steps: int = 800):
    """Everything a convertible trader reads on a screen."""
    v, e, d, dV_dS, gamma = price(cb, mk, steps, split=True)
    parity = cb.ratio * mk.spot
    floor = bond_floor(cb, mk)
    # central differences on wide bumps: a bump changes the tree's grid, and a
    # wider bump keeps that grid noise small relative to the move
    bump = lambda **kw: price(cb, replace(mk, **kw), steps)
    vega = (bump(vol=mk.vol + 0.02) - bump(vol=mk.vol - 0.02)) / 4          # per vol point
    rho = (bump(rate=mk.rate + 0.0005) - bump(rate=mk.rate - 0.0005)) / 10   # per bp
    cs01 = (bump(spread=mk.spread + 0.0005) - bump(spread=mk.spread - 0.0005)) / 10
    return {
        "price": v, "equity_part": e, "debt_part": d,
        "parity": parity, "bond_floor": floor,
        "conversion_premium": v / parity - 1,
        "investment_premium": v / floor - 1,
        "delta": dV_dS / cb.ratio,              # share of parity moves, 0 to 1
        "delta_shares": dV_dS,                  # shares to short per bond
        "gamma": gamma * mk.spot / 100 / cb.ratio,   # change in delta for a 1% spot move
        "vega": vega, "rho": rho, "cs01": cs01,
    }


def implied_vol(cb: Convertible, mk: Market, market_price: float, steps: int = 400,
                lo: float = 0.01, hi: float = 1.5):
    """Volatility that reprices the convertible to its market price."""
    f = lambda s: price(cb, replace(mk, vol=s), steps) - market_price
    if f(lo) > 0 or f(hi) < 0:
        raise ValueError("market price outside the range the model can reach")
    for _ in range(60):
        mid = (lo + hi) / 2
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def hedged_pnl(cb: Convertible, mk: Market, moves=None, steps: int = 800):
    """Convertible arbitrage: long one bond, short its delta in shares.

    Instantaneous P&L for spot moves - the gamma the desk is long.
    """
    if moves is None:
        moves = np.linspace(-0.30, 0.30, 25)
    v0 = price(cb, mk, steps)
    a = analytics(cb, mk, steps)
    out = []
    for m in moves:
        s1 = mk.spot * (1 + m)
        v1 = price(cb, replace(mk, spot=s1), steps)
        out.append((m, (v1 - v0) - a["delta_shares"] * (s1 - mk.spot)))
    return out


def load_underlying(ticker: str):
    """Optional: last close and one-year realised volatility from Yahoo Finance."""
    import yfinance as yf
    h = yf.Ticker(ticker).history(period="1y")["Close"]
    if h.empty:
        raise ValueError(f"no data for {ticker}")
    lr = np.diff(np.log(h.to_numpy()))
    return float(h.iloc[-1]), float(lr.std(ddof=1) * sqrt(252)), str(h.index[-1].date())
