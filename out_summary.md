# Convertible bond pricer - results

**Inputs:** Explicit inputs (illustrative convertible and market)

## Terms

| Term | Value |
|---|---|
| Face / maturity | 100 / 5 years |
| Coupon | 1.00% p.a., 2 per year |
| Conversion price / ratio | 135.00 / 0.7407 shares |
| Issuer call | from year 3.0 at 100, soft call trigger 130% of face |
| Investor put | none |
| Spot / vol | 100.00 / 30.00% |
| Rate / dividend / credit spread | 2.50% / 1.50% / 250 bp |

## Screen

| Measure | Value |
|---|---|
| Price | 97.09 |
| Equity part / debt part (TF split) | 38.27 / 58.82 |
| Parity | 74.07 |
| Bond floor | 82.25 |
| Conversion premium | 31.08% |
| Premium over bond floor | 18.05% |
| Delta (share of parity) | 52.63% |
| Shares to short per bond | 0.3898 |
| Gamma (delta change per 1% spot) | 0.60 pts |
| Vega (per vol point) | 0.581 |
| Rho (per bp) | -0.0277 |
| CS01 (per bp of credit spread) | -0.0284 |

## What each clause is worth

| Structure | Price | vs base |
|---|---|---|
| No issuer call | 97.26 | +0.17 |
| Base: soft call from year 3.0 | 97.09 | - |
| Hard call from year 3.0 | 94.13 | -2.97 |
| Base + investor put at year 3 | 99.58 | +2.49 |

## Implied volatility

A bond quoted at 96.09, one point under model, implies a volatility of 28.32% against 30.00% used: the market is pricing the option 1.7 vol points cheaper (or the credit wider).

## Scenarios (price, spot move x credit spread)

| spread | -30% | -20% | -10% | +0% | +10% | +20% | +30% |
|---|---|---|---|---|---|---|---|
| 100 bp | 93.02 | 95.40 | 98.25 | 101.50 | 105.31 | 109.43 | 114.06 |
| 250 bp | 87.55 | 90.25 | 93.44 | 97.05 | 101.24 | 105.67 | 110.61 |
| 400 bp | 82.47 | 85.46 | 88.98 | 92.92 | 97.46 | 102.19 | 107.41 |
| 600 bp | 76.28 | 79.62 | 83.53 | 87.87 | 92.86 | 97.97 | 103.55 |
| 900 bp | 68.06 | 71.89 | 76.34 | 81.27 | 86.88 | 92.56 | 98.74 |
