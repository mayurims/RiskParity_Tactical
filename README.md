# Multi-Asset Risk Parity with Regime-Conditional Tactical Tilts

**MGMTMFE 431 – Quantitative Asset Management | UCLA Anderson, Spring 2026**  

---

## Overview

This repository implements a two-layer multi-asset portfolio strategy that separates robust risk budgeting from return timing:

**Layer 1 – Equal Risk Contribution (ERC) base**: Allocates capital across seven asset classes so each contributes equally to total portfolio variance. Requires no return forecasts; uses regime-conditional EWMA covariance with Ledoit-Wolf shrinkage.

**Layer 2 – Regime-conditional tactical overlay**: Tilts weights using combination forecasts (12 signals per asset, two regimes) that are shrunk 70% toward zero, gated by a confidence filter, and scaled continuously by a financial conditions score.

**Out-of-sample results (July 1995 – December 2024, 354 months):**

| Portfolio | Sharpe | Max DD | Calmar | Alpha (%) | t-stat |
|---|---|---|---|---|---|
| **ERC + Tactical (ours)** | **0.884** | **-12.2%** | **0.384** | **1.344** | **1.95** |
| ERC base (7 assets) | 0.867 | -12.9% | 0.352 | 1.173 | 1.72 |
| 60/40 | 0.724 | -30.7% | 0.227 | 0.000 | — |
| Vol-Targeted 60/40 | 0.706 | -34.6% | 0.212 | -0.237 | -0.39 |
| Equal-weighted (1/N) | 0.782 | -20.9% | 0.251 | 1.005 | 1.22 |

> Net Sharpe at 10 bps one-way transaction cost: **0.877**. Net Calmar: **0.380**.  
> Alpha estimated against the 8-factor model (FF5 + MOM + TERM + DEF) with Newey-West SEs (6 lags).

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
- **Regime-conditional EWMA**: λ = 0.90 (turbulent) / 0.96 (calm), interpolated continuously by regime score
- **Ledoit-Wolf shrinkage** toward a scaled identity, estimated on a rolling 60-month window
- **Continuous blending** between EWMA and identity components

### Layer 2: Regime-Conditional Tactical Overlay

**Financial Conditions Index (FCI):**

$$\text{FCI}_t = z(\text{credit spread}_t) + z(\text{svar}_t) - z(\text{term spread}_t)$$

All z-scores use expanding windows (no lookahead). The **regime score** $s_t = \Pr(\text{FCI}_s < \text{FCI}_t,\ s < t)$ is a continuous percentile rank — avoiding the cliff-edge flips of a binary flag.

**Combination forecasts**: For each asset $i$ and each of 12 signals $j$ (7 Goyal-Welch predictors, 3 macro predictors, backwardation score, TSMOM), separate OLS regressions are fit within calm and turbulent regimes using all historical regime-matched observations. Forecasts are equal-weighted across signals (Rapach et al., 2010).

**Overlay rule** (three steps):
1. **Shrinkage**: $\tilde{\mu}_{i,t} = 0.30 \times \hat{r}^{\text{combo}}_{i,t+1}$
2. **Confidence gate**: tilt only if $|\tilde{\mu}_{i,t}| > \sigma^{\text{expand}}_{\hat{r}_i}$
3. **Sign-cap tilt scaled by regime**: $w_{i,t} = w^{\text{ERC}}_{i,t} + s_t \cdot \text{sign}(\tilde{\mu}^{\text{gated}}_{i,t}) \times 0.10$, clipped to $[-w^{\text{ERC}}_{i,t},\ 0.10]$

---

## Repository Structure

```
RiskParity_Tactical/
├── data/                 # Raw data files (gitignored — see Download Data below)
├── src/
│   └── project.py        # Full strategy: data loading, ERC, regime scoring,
│                         # forecasting, overlay, backtest, tables, figures
├── results/
│   ├── figures/          # Generated figures (gitignored, regenerated on run)
│   └── tables/           # Generated CSV tables (gitignored, regenerated on run)
├── docs/                 # Project writeup (PDF)
├── requirements.txt
└── README.md
```

**Generated outputs** (after running `project.py`):

| File | Description |
|---|---|
| `fig1_cumulative.png` | Log cumulative excess returns vs benchmarks |
| `fig2_drawdowns.png` | Drawdown series across all portfolios |
| `fig3_rolling_sharpe.png` | Rolling 36-month Sharpe ratio |
| `fig4_weights_heatmap.png` | Weight evolution heatmap (1995–2024) |
| `fig5_rolling_alpha.png` | Rolling 60-month 8-factor alpha |
| `fig6_regime_score.png` | Continuous FCI regime score |
| `table1_performance.csv` | Headline OOS performance metrics |
| `table2_ir.csv` | Information ratios vs benchmarks |
| `table3_alpha.csv` | 8-factor alpha regressions |
| `table4_regime.csv` | Returns by calm/turbulent regime |
| `table5_turnover.csv` | Turnover and net-of-cost Sharpe |
| `table6_tc_sensitivity.csv` | Transaction cost sensitivity |
| `table7_ablation.csv` | Design ablation (Steps A→B→C) |
| `table8_directional.csv` | Directional accuracy by regime |
| `table9_oos_r2.csv` | Campbell-Thompson OOS R² by asset |
| `table10_net_calmar.csv` | Net Calmar ratios at 10 bps |

---

## Getting Started

### 1. Clone and install dependencies

```bash
git clone https://github.com/mayurims/RiskParity_Tactical.git
cd RiskParity_Tactical
pip install -r requirements.txt
```

### 2. Download data

Place all raw data files directly in the `data/` folder. Required sources:

| File | Source | URL |
|---|---|---|
| `F-F_Research_Data_5_Factors_2x3.csv` | Ken French | https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html |
| `Developed_ex_US_5_Factors.csv` | Ken French | https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html |
| `F-F_Momentum_Factor.csv` | Ken French | https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html |
| `goyal_welch_predictors.xlsx` | Goyal-Welch | https://sites.google.com/view/agoyal145 |
| `aqr_commodities.xlsx` | AQR | https://www.aqr.com/Insights/Datasets |
| `GS5.csv` | FRED | https://fred.stlouisfed.org/series/GS5 |
| `GS10.csv` | FRED | https://fred.stlouisfed.org/series/GS10 |
| `INDPRO.csv` | FRED | https://fred.stlouisfed.org/series/INDPRO |
| `CPIAUCSL.csv` | FRED | https://fred.stlouisfed.org/series/CPIAUCSL |

### 3. Run the full pipeline

```bash
python src/project.py
```

This will run the complete pipeline in sequence: data loading → regime scoring → ERC weights → combination forecasts → tactical overlay → backtest → all tables and figures saved to `results/`.

**Note:** The forecasting step runs OLS regressions for each asset × predictor × regime combination across 354 OOS months. Expect a runtime of 15–45 minutes depending on your machine.

---

## Lookahead Bias Controls

All predictors are lagged before use:

| Predictor | Lag applied | Reason |
|---|---|---|
| Goyal-Welch predictors (7) | `.shift(1)` | Observable at end of month t-1 |
| INDPRO (IP momentum) | `.shift(2)` | Publication delay |
| CPI inflation | `.shift(2)` | Publication delay |
| Yield curve slope, backwardation | `.shift(1)` | Observable at end of month t-1 |
| FCI regime score | Expanding percentile over $\{s < t\}$ only | No future FCI values used |
| Confidence gate threshold | Expanding std of past forecasts only | No future forecast values used |

No parameter was searched in-sample. All hyperparameters are set from prior literature before examining OOS performance:

| Parameter | Value | Source |
|---|---|---|
| Forecast shrinkage | 0.30 | Campbell & Thompson (2008) |
| EWMA λ (calm) | 0.96 | Standard EWMA literature |
| EWMA λ (turbulent) | 0.90 | Adrian et al. (2019) |
| Tilt cap | ±10 pp | Symmetry argument |
| Confidence gate | 1.0σ | Symmetry argument |
| Burn-in period | 60 months (Jul 1990 – Jun 1995) | Sufficient history for covariance and regime estimation |

---

## Key References

- Adrian, T., Boyarchenko, N., and Giannone, D. (2019). Vulnerable growth. *AER*, 109(4), 1263–1289.
- Campbell, J. Y. and Thompson, S. B. (2008). Predicting excess stock returns out of sample. *RFS*, 21(4), 1509–1531.
- DeMiguel, V., Garlappi, L., and Uppal, R. (2009). Optimal versus naive diversification. *RFS*, 22(5), 1915–1953.
- Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *JMVA*, 88(2), 365–411.
- Levine, A., Ooi, Y. H., Richardson, M., and Sasseville, C. (2018). Commodities for the long run. *FAJ*, 74(2), 55–68.
- Maillard, S., Roncalli, T., and Teiletche, J. (2010). The properties of equally weighted risk contribution portfolios. *JPM*, 36(4), 60–70.
- Rapach, D. E., Strauss, J. K., and Zhou, G. (2010). Out-of-sample equity premium prediction: Combination forecasts. *RFS*, 23(2), 821–862.

---