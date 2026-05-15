# Multi-Asset Risk Parity with Regime-Conditional Tactical Tilts

**MGMTMFE 431 – Quantitative Asset Management | UCLA Anderson, Spring 2026**  

---

## Overview

This repository implements a two-layer multi-asset portfolio strategy that separates robust risk budgeting from return timing:

1. **Layer 1 – Equal Risk Contribution (ERC) base**: Allocates capital across seven asset classes so each contributes equally to total portfolio variance. Requires no return forecasts; uses regime-conditional EWMA covariance with Ledoit-Wolf shrinkage.

2. **Layer 2 – Regime-conditional tactical overlay**: Tilts weights using combination forecasts (12 predictors per asset, two regimes) that are shrunk 70% toward zero, gated by a confidence filter, and scaled continuously by a financial conditions score.

**Out-of-sample results (July 1995 – December 2024, 354 months):**

| Portfolio | Sharpe | Max DD | Calmar | Alpha (%) | t-stat |
|---|---|---|---|---|---|
| **ERC + Tactical (ours)** | **0.884** | **-12.2%** | **0.384** | **1.344** | **1.95** |
| ERC base (7 assets) | 0.867 | -12.9% | 0.352 | 1.173 | 1.72 |
| 60/40 | 0.724 | -30.7% | 0.227 | 0.000 | — |
| Equal-weighted (1/N) | 0.782 | -20.9% | 0.251 | 1.005 | 1.22 |

> Net Sharpe at 10 bps one-way transaction cost: **0.877**. Net Calmar: **0.380**.

---

## Asset Universe

| ID | Description | Source |
|---|---|---|
| `US_EQ` | US equity market excess return (Mkt-RF) | Ken French |
| `INTL_EQ` | Developed ex-US equity excess return | Ken French |
| `LONG_TREAS` | Long-term US Treasury excess return | Goyal-Welch |
| `CREDIT` | IG corporate bond excess return | Goyal-Welch |
| `COMMOD` | Equal-weight commodity excess return | AQR |
| `INT_BOND` | 5-year Treasury excess return (duration approx.) | FRED GS5 |
| `LS_COMMOD` | Long/short commodity carry excess return | AQR |

The `LS_COMMOD` asset (Levine et al., 2018) has Sharpe 0.60, correlation 0.035 with US equity, and correlation -0.04 with the long-only commodity return — making it near-orthogonal to all standard risk factors and the primary source of 8-factor alpha in this strategy.

---

## Methodology

### Layer 1: ERC Base Portfolio

Weights solve the log-barrier dual (Maillard et al., 2010):

$$\min_{y > 0} \frac{1}{2} y^\top \Sigma_t y - \mathbf{1}^\top \log y, \quad w = y / \mathbf{1}^\top y$$

**Covariance estimation** uses three enhancements over a plain sample covariance:
- **Regime-conditional EWMA**: λ = 0.90 (turbulent) / 0.96 (calm), interpolated by regime score
- **Ledoit-Wolf shrinkage** toward a scaled identity, estimated on a rolling 60-month window
- **Continuous blending** between EWMA and identity components

### Layer 2: Regime-Conditional Tactical Overlay

**Financial Conditions Index (FCI):**

$$\text{FCI}_t = z(\text{credit spread}_t) + z(\text{svar}_t) - z(\text{term spread}_t)$$

All z-scores use expanding windows (no lookahead). The **regime score** $s_t = \Pr(\text{FCI}_s < \text{FCI}_t,\ s < t)$ is a continuous percentile rank — avoiding cliff-edge flips of binary flags.

**Combination forecasts**: For each asset $i$ and each of 12 predictors $j$ (7 Goyal-Welch, 3 macro, backwardation, TSMOM), separate OLS regressions are fit within calm and turbulent regimes. Forecasts are equal-weighted across predictors (Rapach et al., 2010).

**Overlay rule** (three steps):
1. **Shrinkage**: $\tilde{\mu}_{i,t} = 0.30 \times \hat{r}^{\text{combo}}_{i,t+1}$
2. **Confidence gate**: tilt only if $|\tilde{\mu}_{i,t}| > \sigma^{\text{expand}}_{\hat{r}_i}$
3. **Sign-cap tilt scaled by regime**: $w_{i,t} = w^{\text{ERC}}_{i,t} + s_t \cdot \text{sign}(\tilde{\mu}^{\text{gated}}_{i,t}) \times 0.10$, clipped to $[-w^{\text{ERC}}_{i,t},\ 0.10]$

---

## Repository Structure

```
risk-parity-tactical/
├── data/                 # Raw downloaded data (gitignored)
│
├── src/
│   ├── project.py
|
├── results/
│   ├── figures/          # All paper figures (gitignored, regenerated)
│   └── tables/           # All paper tables (gitignored, regenerated)
│
├── docs/                 # Project Writeup
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone and install dependencies

```bash
git clone https://github.com/<your-username>/risk-parity-tactical.git
cd risk-parity-tactical
pip install -r requirements.txt
```

### 2. Download data

Place raw data files in `data/raw/`. Required sources:

| Source | URL |
|---|---|
| Ken French Data Library | https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html |
| Goyal-Welch dataset | https://sites.google.com/view/agoyal145 |
| AQR Data Library | https://www.aqr.com/Insights/Datasets |
| FRED (GS5, GS10, INDPRO, CPI) | https://fred.stlouisfed.org |

### 3. Run the backtest

```bash
python -m src.utils.data_loader        # Build processed panel
python -m src.utils.backtest           # Run OOS backtest (Jul 1995 – Dec 2024)
```

---

## Lookahead Bias Controls

All predictors are lagged before use:
- Goyal-Welch predictors: `.shift(1)`
- INDPRO and CPI: `.shift(2)` (publication delay)
- Yield curve slope, backwardation: `.shift(1)`
- FCI regime score: expanding percentile using only $\{\text{FCI}_s : s < t\}$
- Confidence gate threshold: expanding standard deviation of *past* forecasts only

No parameter in the strategy was searched in-sample. All hyperparameters (shrinkage 0.30, EWMA lambdas 0.90/0.96, tilt cap ±10 pp, gate multiplier 1.0σ) are set from prior literature before examining OOS performance.

---

## Key References

- Adrian, T., Boyarchenko, N., and Giannone, D. (2019). Vulnerable growth. *AER*, 109(4), 1263–1289.
- Campbell, J. Y. and Thompson, S. B. (2008). Predicting excess stock returns out of sample. *RFS*, 21(4), 1509–1531.
- Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *JMVA*, 88(2), 365–411.
- Levine, A., Ooi, Y. H., Richardson, M., and Sasseville, C. (2018). Commodities for the long run. *FAJ*, 74(2), 55–68.
- Maillard, S., Roncalli, T., and Teiletche, J. (2010). The properties of equally weighted risk contribution portfolios. *JPM*, 36(4), 60–70.
- Rapach, D. E., Strauss, J. K., and Zhou, G. (2010). Out-of-sample equity premium prediction: Combination forecasts. *RFS*, 23(2), 821–862.

---

## License

For academic use. Please cite the original paper if you build on this work.
