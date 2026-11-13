from dataclasses import replace
from math import exp

import pytest

from cb_pricer import (Convertible, Market, price, analytics, bond_floor, bs_call,
                       implied_vol, hedged_pnl)

CB, MK = Convertible(), Market()


def test_matches_bond_plus_call_without_credit_or_features():
    """No coupon, no call, no dividend, no spread: conversion is never early,
    so the convertible is a zero-coupon bond plus ratio x European call."""
    cb = Convertible(coupon=0.0, call_start=None)
    mk = Market(spread=0.0, div=0.0)
    ref = cb.face * exp(-mk.rate * cb.maturity) + cb.ratio * bs_call(
        mk.spot, cb.conv_price, cb.maturity, mk.rate, 0.0, mk.vol)
    assert price(cb, mk) == pytest.approx(ref, abs=0.03)


def test_price_above_parity_and_bond_floor_without_call():
    cb = replace(CB, call_start=None)
    for s in (40, 100, 160):
        mk = replace(MK, spot=s)
        v = price(cb, mk)
        assert v >= cb.ratio * s - 1e-9
        assert v >= bond_floor(cb, mk) - 0.05


def test_deep_in_the_money_behaves_like_shares():
    a = analytics(replace(CB, call_start=None), replace(MK, spot=400))
    assert a["conversion_premium"] < 0.01
    assert a["delta"] > 0.97


def test_deep_out_of_the_money_behaves_like_the_bond():
    a = analytics(CB, replace(MK, spot=10))
    assert a["price"] == pytest.approx(a["bond_floor"], abs=0.05)
    assert a["delta"] < 0.01


def test_clauses_move_the_price_the_right_way():
    base = price(CB, MK)
    assert price(replace(CB, call_start=None), MK) >= base - 1e-9    # call belongs to the issuer
    assert price(replace(CB, call_trigger=0.0), MK) < base           # a hard call is worth more to it
    assert price(replace(CB, put_date=3.0), MK) > base               # put belongs to the holder


def test_sensitivity_signs():
    a = analytics(replace(CB, call_start=None), MK)
    assert 0 < a["delta"] < 1
    assert a["gamma"] > 0
    assert a["vega"] > 0
    assert a["cs01"] < 0


def test_equity_plus_debt_equals_price():
    v, e, d, _, _ = price(CB, MK, split=True)
    assert e + d == pytest.approx(v, abs=1e-10)


def test_implied_vol_round_trip():
    v = price(CB, replace(MK, vol=0.27), 400)
    assert implied_vol(CB, MK, v, steps=400) == pytest.approx(0.27, abs=0.003)


def test_delta_hedged_position_is_long_gamma():
    pnl = dict(hedged_pnl(replace(CB, call_start=None), MK, [-0.2, 0.0, 0.2]))
    assert pnl[-0.2] > 0 and pnl[0.2] > 0
    assert pnl[0.0] == pytest.approx(0.0, abs=1e-9)
