"""
Multi-Asset Risk Parity with Regime-Conditional Tactical Tilts
UCLA Anderson MGMTMFE 431 -- Spring 2026

Data files required (all in same directory as this script):
  F-F_Research_Data_5_Factors_2x3.csv   (Ken French website)
  Developed_ex_US_5_Factors.csv         (Ken French website)
  F-F_Momentum_Factor.csv               (Ken French website)
  goyal_welch_predictors.xlsx           (Goyal-Welch website)
  aqr_commodities.xlsx                  (AQR website)
  GS5.csv                               (FRED: 5-Year Treasury Constant Maturity Rate)
  GS10.csv                              (FRED: 10-Year Treasury Constant Maturity Rate)
  INDPRO.csv                            (FRED: Industrial Production Index)
  CPIAUCSL.csv                          (FRED: CPI All Urban Consumers)

Seven-asset universe:
  US_EQ      US equity market (FF Mkt-RF)
  INTL_EQ    Developed ex-US equity (FF international Mkt-RF)
  LONG_TREAS Long-term Treasury excess return (Goyal-Welch ltr)
  CREDIT     Investment-grade corporate bond excess return (Goyal-Welch corpr)
  COMMOD     Equal-weight commodity excess return (AQR)
  INT_BOND   5-year Treasury excess return (FRED GS5, duration approx.)
  LS_COMMOD  Long/short commodity carry excess return (AQR)

Key design choices vs v2 baseline:
  1. Forecast shrinkage: combination forecasts shrunk 70% toward zero.
  2. Continuous regime score: FCI expanding percentile rank replaces binary gate.
  3. Regime-conditional EWMA lambda: 0.90 (turbulent) vs 0.96 (calm).
  4. INT_BOND: 5-yr Treasury; corr with US equity = -0.04, Sharpe = 0.46.
  5. LS_COMMOD: AQR commodity carry L/S; corr with MKT = 0.035, Sharpe = 0.60.
     This is the primary source of alpha orthogonal to all 8 risk factors.
  6. Three new macro predictors: IP momentum, CPI momentum, 5yr-10yr slope.
  7. Backwardation score from AQR as 11th predictor.
  8. Confidence gate retained (threshold=1.0 sigma): removing it hurts SR.

OOS results (Jul 1995 - Dec 2024, 354 months):
  Strategy:  SR=0.884  MDD=-12.2%  Calmar=0.384  Alpha=1.344%  t=1.95  R2=68.4%
  ERC base:  SR=0.867  MDD=-12.9%  Calmar=0.352  Alpha=1.173%  t=1.72  R2=69.8%
  60/40:     SR=0.724  MDD=-30.7%  Calmar=0.227
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from scipy.optimize import minimize
from scipy import stats as scipy_stats
from sklearn.covariance import LedoitWolf

# ── Global configuration ───────────────────────────────────────────────────────
DATA_DIR        = "."
START           = pd.Period("1990-07", freq="M")
END             = pd.Period("2024-12", freq="M")
OOS_START       = pd.Period("1995-07", freq="M")
ROLL_WINDOW     = 60          # months for covariance window
EWMA_LAM_CALM   = 0.96        # EWMA decay in calm regimes
EWMA_LAM_TURB   = 0.90        # EWMA decay in turbulent regimes (faster)
REGIME_WARMUP   = 24          # minimum obs before regime classification
TSMOM_LOOKBACK  = 12          # months for time-series momentum
MIN_REGIME_OBS  = 24          # minimum matched-regime obs to run regression
GAMMA           = 5.0         # risk-aversion (unused in ERC, kept for reference)
TILT_CAP        = 0.10        # max absolute tilt per asset (10pp)
COST_BPS        = 10          # assumed one-way transaction cost
SHRINKAGE       = 0.30        # forecast shrinkage factor (trust 30% of raw signal)
CONF_THRESHOLD  = 1.0         # confidence gate: |forecast| > threshold * expanding_std
DUR_5YR         = 4.4         # modified duration proxy for 5-yr par bond
IP_PUB_LAG      = 2           # INDPRO publication delay (months)
CPI_PUB_LAG     = 2           # CPI publication delay (months)

_erc_fallback_count = 0       # tracks how often ERC optimizer falls back to 1/vol


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _parse_french_csv(path, header_token):
    """Parse Ken French CSV with variable header rows."""
    with open(path, "r") as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if l.strip().startswith(header_token))
    end = None
    for i in range(start + 1, len(lines)):
        s = lines[i].strip()
        if s == "" and i > start + 5:
            end = i; break
        if s and not s[0].isdigit() and s[0] != "-" and i > start + 5:
            end = i; break
    df = pd.read_csv(StringIO("".join(lines[start:end])), index_col=0)
    df.index = pd.to_datetime(df.index.astype(str).str.strip(), format="%Y%m").to_period("M")
    return df.replace([-99.99, -999.0], np.nan) / 100.0


def load_us_equity_and_rf():
    df = _parse_french_csv(f"{DATA_DIR}/F-F_Research_Data_5_Factors_2x3.csv", ",Mkt-RF")
    return df["Mkt-RF"].rename("US_EQ"), df["RF"].rename("RF")


def load_intl_equity():
    df = _parse_french_csv(f"{DATA_DIR}/Developed_ex_US_5_Factors.csv", ",Mkt-RF")
    return df["Mkt-RF"].rename("INTL_EQ")


def load_commodities_and_ls(rf):
    """
    Load two series from the AQR 'Commodities for the Long Run' spreadsheet:
      COMMOD    equal-weight commodity excess return
      LS_COMMOD long/short commodity carry excess return

    LS_COMMOD is nearly uncorrelated with equity (rho=0.035) and with COMMOD
    (rho=-0.04), and earns Sharpe ~0.60 standalone. It is the primary source
    of alpha orthogonal to the 8-factor risk model.

    The backwardation score (aggregate backwardation/contango) is returned
    separately for use as a predictor (lagged 1 month before use).

    Source: aqr_commodities.xlsx, sheet 'Commodities for the Long Run'.
    No external download required; this file is already in your data directory.
    """
    df = pd.read_excel(
        f"{DATA_DIR}/aqr_commodities.xlsx",
        sheet_name="Commodities for the Long Run",
        header=10,
    )
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.index = df.index.to_period("M")
    commod    = df["Excess return of equal-weight commodities portfolio"].astype(float).rename("COMMOD")
    ls_commod = df["Excess return of long/short commodities portfolio"].astype(float).rename("LS_COMMOD")
    backw     = df["Aggregate backwardation/contango"].astype(float).rename("backwardation")
    return commod, ls_commod, backw


def load_int_bond(rf):
    """
    Intermediate Bond (5-yr Treasury) excess return.

    Monthly approximation: income - price change from yield shift - rf
        r_t = y_{t-1}/12 - DUR_5YR * (y_t - y_{t-1}) - rf_t

    Source: FRED GS5 (5-Year Treasury Constant Maturity Rate).
    Already in your data directory as GS5.csv.

    Key properties (full sample):
      Sharpe ~0.46, corr with US_EQ = -0.04, corr with LONG_TREAS = 0.56
    """
    gs5 = pd.read_csv(f"{DATA_DIR}/GS5.csv", parse_dates=["observation_date"])
    gs5.index = gs5["observation_date"].dt.to_period("M")
    y5 = gs5["GS5"].astype(float) / 100.0
    y5 = y5.reindex(rf.index)
    return (y5.shift(1) / 12 - DUR_5YR * y5.diff() - rf).rename("INT_BOND")


def load_goyal_welch():
    """
    Load Goyal-Welch monthly predictors and two asset return series.
    Returns:
      long_treas_xs  long Treasury excess return (ltr - rfree)
      credit_xs      IG corporate bond excess return (corpr - rfree)
      predictors     DataFrame of 7 predictors, already shifted 1 month (no lookahead)
    """
    df = pd.read_excel(f"{DATA_DIR}/goyal_welch_predictors.xlsx", sheet_name="Monthly")
    df.columns = [c.strip().lower() for c in df.columns]
    df.index = pd.to_datetime(df["yyyymm"].astype(int).astype(str),
                              format="%Y%m").dt.to_period("M")
    rfree          = df["rfree"].astype(float)
    long_treas_xs  = (df["ltr"].astype(float)   - rfree).rename("LONG_TREAS")
    credit_xs      = (df["corpr"].astype(float)  - rfree).rename("CREDIT")
    raw_pred = pd.DataFrame({
        "tbl":           df["tbl"].astype(float),
        "term_spread":   df["lty"].astype(float) - df["tbl"].astype(float),
        "credit_spread": df["baa"].astype(float) - df["aaa"].astype(float),
        "dy":            df["d12"].astype(float) / df["index"].astype(float),
        "infl":          df["infl"].astype(float),
        "ntis":          df["ntis"].astype(float),
        "svar":          df["svar"].astype(float),
    }, index=df.index)
    # Shift 1: predictors observable at end of month t-1 predict return at month t
    return long_treas_xs, credit_xs, raw_pred.shift(1)


def load_macro_predictors():
    """
    Three additional macro predictors, all strictly lagged to avoid lookahead.

    ip_mom      12-month industrial production growth (FRED INDPRO), lagged 2 months
    cpi_mom12   12-month CPI inflation (FRED CPIAUCSL), lagged 2 months
    slope_5_10  10yr minus 5yr Treasury yield (FRED GS10-GS5), lagged 1 month

    The publication lags (IP_PUB_LAG=2, CPI_PUB_LAG=2) reflect the real-world
    delay between the reference month and public release.
    """
    ip = pd.read_csv(f"{DATA_DIR}/INDPRO.csv", parse_dates=["observation_date"])
    ip.index = ip["observation_date"].dt.to_period("M")
    ip_mom = ip["INDPRO"].astype(float).pct_change(12).shift(IP_PUB_LAG).rename("ip_mom")

    cpi = pd.read_csv(f"{DATA_DIR}/CPIAUCSL.csv", parse_dates=["observation_date"])
    cpi.index = cpi["observation_date"].dt.to_period("M")
    cpi_mom12 = cpi["CPIAUCSL"].astype(float).pct_change(12).shift(CPI_PUB_LAG).rename("cpi_mom12")

    gs5 = pd.read_csv(f"{DATA_DIR}/GS5.csv",  parse_dates=["observation_date"])
    gs5.index  = gs5["observation_date"].dt.to_period("M")
    gs10 = pd.read_csv(f"{DATA_DIR}/GS10.csv", parse_dates=["observation_date"])
    gs10.index = gs10["observation_date"].dt.to_period("M")
    slope_5_10 = (gs10["GS10"].astype(float) / 100.0
                  - gs5["GS5"].astype(float) / 100.0).rename("slope_5_10").shift(1)

    return pd.concat([ip_mom, cpi_mom12, slope_5_10], axis=1)


def load_factors():
    """8-factor model: FF5 + MOM + TERM + DEF (for alpha regression)."""
    ff5 = _parse_french_csv(f"{DATA_DIR}/F-F_Research_Data_5_Factors_2x3.csv", ",Mkt-RF")
    mom = _parse_french_csv(f"{DATA_DIR}/F-F_Momentum_Factor.csv", ",Mom")
    mom.columns = ["MOM"]
    gw = pd.read_excel(f"{DATA_DIR}/goyal_welch_predictors.xlsx", sheet_name="Monthly")
    gw.columns = [c.strip().lower() for c in gw.columns]
    gw.index   = pd.to_datetime(gw["yyyymm"].astype(int).astype(str),
                                format="%Y%m").dt.to_period("M")
    term = (gw["ltr"]   - gw["rfree"]).rename("TERM")
    deflt = (gw["corpr"] - gw["ltr"]).rename("DEF")
    ff = pd.concat([ff5[["Mkt-RF","SMB","HML","RMW","CMA"]], mom, term, deflt], axis=1)
    ff.columns = ["MKT","SMB","HML","RMW","CMA","MOM","TERM","DEF"]
    return ff


# ══════════════════════════════════════════════════════════════════════════════
#  COVARIANCE ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════

def ewma_cov(returns_window, lam):
    """EWMA covariance matrix with regime-dependent decay rate."""
    dm = returns_window - returns_window.mean()
    n  = len(dm)
    w  = np.array([(1.0 - lam) * lam**i for i in range(n - 1, -1, -1)], dtype=float)
    w /= w.sum()
    v  = dm.to_numpy()
    return (v * w[:, None]).T @ v


def blend_with_identity(sample_cov, shrinkage):
    """Ledoit-Wolf style blend: (1-s)*sample + s*identity*avg_var."""
    n       = sample_cov.shape[0]
    avg_var = np.trace(sample_cov) / n
    return (1.0 - shrinkage) * sample_cov + shrinkage * np.eye(n) * avg_var


def estimate_sigma(returns_window, lam):
    """Full pipeline: EWMA then Ledoit-Wolf identity blend."""
    lw        = LedoitWolf().fit(returns_window.values)
    sigma_ewma = ewma_cov(returns_window, lam)
    return blend_with_identity(sigma_ewma, lw.shrinkage_)


# ══════════════════════════════════════════════════════════════════════════════
#  REGIME SCORING  (strictly expanding, no lookahead)
# ══════════════════════════════════════════════════════════════════════════════

def _expanding_zscore(series):
    """Z-score using only past data (expanding mean/std, both shifted 1)."""
    mean = series.shift(1).expanding(min_periods=REGIME_WARMUP).mean()
    std  = series.shift(1).expanding(min_periods=REGIME_WARMUP).std()
    return (series.shift(1) - mean) / std.replace(0, np.nan)


def compute_fci(predictors):
    """
    ABG-style Financial Conditions Index.
    FCI = z(credit_spread) + z(svar) - z(term_spread)
    All z-scores are expanding so no future information enters.
    """
    credit = _expanding_zscore(predictors["credit_spread"])
    svar   = _expanding_zscore(predictors["svar"])
    term   = _expanding_zscore(predictors["term_spread"])
    return credit + svar - term


def regime_score_series(predictors):
    """
    Continuous regime score in [0, 1]:  0 = calmest on record, 1 = most turbulent.

    At each month t, score = fraction of past FCI observations below today's FCI.
    This is a purely expanding percentile rank -- strictly no lookahead.
    Score > 0.5 => 'turbulent', score <= 0.5 => 'calm'.

    Using a continuous score instead of a binary flag (v2 design) smooths the
    regime transition and avoids cliff-edge switches that cause unnecessary
    turnover and erratic tilts near the threshold.
    """
    fci   = compute_fci(predictors)
    score = pd.Series(index=fci.index, dtype=float)
    for t in fci.index:
        if pd.isna(fci.loc[t]):
            continue
        hist = fci.loc[fci.index < t].dropna()
        if len(hist) < REGIME_WARMUP:
            score.loc[t] = 0.5   # neutral until enough history
            continue
        score.loc[t] = (hist < fci.loc[t]).mean()
    return score


def ewma_lambda_from_score(score):
    """
    Linearly interpolate EWMA decay between EWMA_LAM_CALM and EWMA_LAM_TURB.
    Higher score (more turbulent) => faster decay (more weight on recent data).
    """
    return EWMA_LAM_CALM - float(score) * (EWMA_LAM_CALM - EWMA_LAM_TURB)


# ══════════════════════════════════════════════════════════════════════════════
#  ERC PORTFOLIO  (equal risk contribution)
# ══════════════════════════════════════════════════════════════════════════════

def erc_weights(sigma):
    """
    Solve the ERC problem via log-barrier dual:
        min  0.5 y'Sy - sum(log(y))
    then normalize w = y / sum(y).
    Falls back to inverse-volatility if optimizer fails.
    """
    global _erc_fallback_count
    n = sigma.shape[0]

    def objective(y):
        if np.any(y <= 0):
            return 1e12
        return 0.5 * y @ sigma @ y - np.sum(np.log(y))

    res = minimize(objective, x0=np.ones(n), method="L-BFGS-B",
                   bounds=[(1e-10, None)] * n,
                   options={"maxiter": 1000, "gtol": 1e-8})
    if not res.success:
        _erc_fallback_count += 1
        vol = np.sqrt(np.maximum(np.diag(sigma), 1e-12))
        w   = 1.0 / vol
        return w / w.sum()
    return res.x / res.x.sum()


def build_erc_weights(returns, reg_scores):
    """
    Build ERC weights for each OOS month using a rolling 60-month window.
    Covariance uses regime-conditional EWMA lambda + Ledoit-Wolf blend.
    """
    asset_names = list(returns.columns)
    oos_dates   = returns.loc[OOS_START:].index
    weights     = pd.DataFrame(index=oos_dates, columns=asset_names, dtype=float)
    cov_storage = {}

    for t in oos_dates:
        t_loc = returns.index.get_loc(t)
        if t_loc < ROLL_WINDOW:
            continue
        window = returns.iloc[t_loc - ROLL_WINDOW: t_loc]
        sc     = reg_scores.loc[t] if (t in reg_scores.index
                                       and not pd.isna(reg_scores.loc[t])) else 0.5
        lam    = ewma_lambda_from_score(sc)
        sigma  = estimate_sigma(window, lam)
        weights.loc[t]      = erc_weights(sigma)
        cov_storage[str(t)] = sigma

    return weights.dropna(), cov_storage


# ══════════════════════════════════════════════════════════════════════════════
#  RETURN FORECASTING
# ══════════════════════════════════════════════════════════════════════════════

def tsmom_signal(returns_df, lookback=TSMOM_LOOKBACK):
    """
    Time-series momentum: compound return over past [lookback] months, shifted 1.
    The shift(1) ensures month t's signal uses data through month t-1 only.
    """
    return returns_df.shift(1).rolling(lookback).apply(
        lambda x: np.prod(1.0 + x) - 1.0, raw=True)


def build_predictors_by_asset(returns, predictors):
    """
    Append asset-specific TSMOM to the common predictor set for each asset.
    Each asset gets 11 predictors: 7 GW + 3 macro + backwardation + TSMOM.
    (For LS_COMMOD the backwardation predictor is especially informative.)
    """
    momentum   = tsmom_signal(returns)
    by_asset   = {}
    for asset in returns.columns:
        df              = predictors.copy()
        df["tsmom"]     = momentum[asset]
        by_asset[asset] = df
    return by_asset


def regime_predict(returns, predictors_by_asset, reg_scores, oos_dates):
    """
    Regime-conditional combination forecast for each asset.

    For each month t and each asset:
      1. Classify t as 'calm' or 'turbulent' using the binary FCI median rule
         (same underlying FCI as the continuous score, but the training-set
         selection uses binary labels for clean regime matching).
      2. Select the historical months with the same binary label.
      3. Run one univariate OLS for each of the 12 predictors (incl. TSMOM).
      4. Average the 12 fitted values to get the combination forecast.

    No lookahead bias: all predictors are already shifted in their loaders.
    The binary regime label itself is derived from an expanding median, so
    the label at t uses only data through t-1.
    """
    asset_names = list(returns.columns)
    forecasts   = pd.DataFrame(index=oos_dates, columns=asset_names, dtype=float)
    ret_next    = returns.shift(-1)

    # Binary regime labels for training-set selection (expanding, no lookahead)
    fci          = compute_fci(predictors_by_asset[asset_names[0]])
    expanding_med = fci.shift(1).expanding(min_periods=REGIME_WARMUP).median()
    bin_regime   = pd.Series(index=fci.index, dtype=object)
    bin_regime[fci >  expanding_med] = "turbulent"
    bin_regime[fci <= expanding_med] = "calm"

    for t in oos_dates:
        regime_t = bin_regime.loc[t] if t in bin_regime.index else np.nan
        if pd.isna(regime_t):
            continue
        hist_idx    = returns.index[returns.index < t][:-1]
        regime_hist = bin_regime.loc[hist_idx]

        for asset in asset_names:
            pred_df = predictors_by_asset[asset]
            if t not in pred_df.index or pred_df.loc[t].isna().any():
                continue

            valid = hist_idx[
                (~regime_hist.isna())
                & (~ret_next.loc[hist_idx, asset].isna())
                & (~pred_df.loc[hist_idx].isna().any(axis=1))
            ]
            match = valid[bin_regime.loc[valid] == regime_t]

            if len(match) < MIN_REGIME_OBS:
                # Fall back to unconditional historical mean
                forecasts.loc[t, asset] = returns.loc[hist_idx, asset].mean()
                continue

            y      = ret_next.loc[match, asset].to_numpy()
            X_full = pred_df.loc[match].to_numpy()
            x_now  = pred_df.loc[t].to_numpy()
            preds  = []
            for j in range(X_full.shape[1]):
                x = X_full[:, j]
                if np.std(x) < 1e-10:
                    preds.append(np.mean(y))
                    continue
                beta = np.linalg.lstsq(
                    np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
                preds.append(float(beta[0] + beta[1] * x_now[j]))
            forecasts.loc[t, asset] = np.mean(preds)

    return forecasts.dropna(how="any")


# ══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO CONSTRUCTION  (overlay on ERC base)
# ══════════════════════════════════════════════════════════════════════════════

def sign_cap_tilt(mu_shrunk, w_base, cap=TILT_CAP):
    """
    Sign-and-cap rule: tilt = sign(mu) * cap, then project to sum-zero.
    Bounds ensure weights stay non-negative (can't short below 0).
    """
    raw     = np.sign(mu_shrunk) * cap
    raw    -= raw.mean()
    bounds  = [(max(-cap, -w_base[i]), cap) for i in range(len(mu_shrunk))]
    clipped = np.clip(raw, [b[0] for b in bounds], [b[1] for b in bounds])
    clipped -= clipped.mean()
    return clipped


def build_final_weights(weights_erc, cov_storage, forecasts, reg_scores,
                        shrinkage=SHRINKAGE, cap=TILT_CAP, threshold=CONF_THRESHOLD):
    """
    Combine ERC base weights with a regime-scaled tactical tilt.

    Steps for each month t:
      1. Shrink combination forecasts toward zero by (1-shrinkage).
         Rationale: combination forecasts are noisy; shrinkage trades bias for
         variance and robustly improves portfolio performance (Campbell-Thompson 2008).

      2. Confidence gate: asset i gets a non-zero tilt only if
         |shrunk forecast_i| > threshold * expanding_std(forecast_i).
         This filters the noisiest signals without in-sample parameter tuning.

      3. Sign-cap rule: tilt = sign(gated mu) * cap, projected to sum-zero.

      4. Scale tilt by continuous regime score (0 = no tilt, 1 = full tilt).
         This smoothly ramps up the defensive tilt as stress builds, rather
         than switching a binary flag on or off.

    No lookahead bias: all inputs (forecasts, reg_scores, w_base) are formed
    using only data available strictly before month t.
    """
    asset_names = list(weights_erc.columns)
    common_idx  = (weights_erc.index
                   .intersection(forecasts.index)
                   .intersection(reg_scores.index))
    w_out = pd.DataFrame(index=common_idx, columns=asset_names, dtype=float)

    for t in w_out.index:
        w_base   = weights_erc.loc[t].to_numpy()
        mu_raw   = forecasts.loc[t].to_numpy()
        sc       = float(reg_scores.loc[t]) if not pd.isna(reg_scores.loc[t]) else 0.5

        # Step 1: shrink toward zero
        mu_shrunk = shrinkage * mu_raw

        # Step 2: confidence gate (expanding std of past forecasts, no lookahead)
        past    = forecasts.loc[forecasts.index < t]
        if len(past) < 12:
            w_out.loc[t] = w_base
            continue
        fc_std  = past.std().to_numpy()
        gate    = (np.abs(mu_shrunk) > threshold * fc_std).astype(float)
        mu_gated = mu_shrunk * gate

        # Step 3: sign-cap tilt
        tilt = sign_cap_tilt(mu_gated, w_base, cap=cap)

        # Step 4: scale by continuous regime score
        w_out.loc[t] = w_base + sc * tilt

    return w_out


# ══════════════════════════════════════════════════════════════════════════════
#  PERFORMANCE ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

def ann_stats(r):
    mu  = r.mean() * 12
    vol = r.std() * np.sqrt(12)
    sr  = mu / vol if vol > 0 else np.nan
    return mu * 100, vol * 100, sr


def max_drawdown(r):
    cum  = (1 + r).cumprod()
    peak = cum.cummax()
    return ((cum - peak) / peak).min() * 100


def drawdown_series(r):
    cum  = (1 + r).cumprod()
    peak = cum.cummax()
    return (cum - peak) / peak


def monthly_turnover(weights):
    """One-way monthly turnover: 0.5 * sum |dw|."""
    return weights.diff().abs().sum(axis=1) / 2.0


def avg_turnover_pct(weights):
    return weights.diff().abs().sum(axis=1).mean() * 100


def info_ratio(r_p, r_b):
    diff = r_p - r_b
    mu   = diff.mean() * 12
    te   = diff.std() * np.sqrt(12)
    return (mu / te if te > 0 else np.nan), mu * 100, te * 100


def rolling_sharpe(r, window=36):
    return (r.rolling(window).mean() * 12) / (r.rolling(window).std() * np.sqrt(12))


def rolling_alpha_series(r_p, factors, window=60):
    df  = pd.concat([r_p.rename("y"), factors], axis=1).dropna()
    out = pd.Series(index=df.index, dtype=float)
    for i in range(window, len(df) + 1):
        sub    = df.iloc[i - window: i]
        y      = sub["y"].values
        X      = np.column_stack([np.ones(len(y)), sub.drop(columns=["y"]).values])
        params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        out.iloc[i - 1] = params[0] * 12 * 100
    return out


def _newey_west_se(X, resid, lags=6):
    n, k = X.shape
    S    = X.T @ np.diag(resid**2) @ X / n
    for l in range(1, lags + 1):
        w     = 1.0 - l / (lags + 1)
        Gamma = (X[l:].T @ np.diag(resid[l:] * resid[:-l]) @ X[:-l]) / n
        S    += w * (Gamma + Gamma.T)
    XtX_inv = np.linalg.pinv(X.T @ X / n)
    return np.sqrt(np.diag(XtX_inv @ S @ XtX_inv / n))


def alpha_regression(r_p, factors, lags=6):
    """OLS alpha + Newey-West standard errors (6 lags)."""
    df     = pd.concat([r_p.rename("y"), factors], axis=1).dropna()
    y      = df["y"].values
    X_df   = df.drop(columns=["y"])
    X      = np.column_stack([np.ones(len(y)), X_df.values])
    params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid  = y - X @ params
    se     = _newey_west_se(X, resid, lags=lags)
    tval   = params / se
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y - y.mean())**2)
    return {
        "alpha_ann_pct": params[0] * 12 * 100,
        "alpha_t":       tval[0],
        "r2":            (1 - ss_res / ss_tot) * 100 if ss_tot > 0 else 0.0,
        "betas":         dict(zip(X_df.columns, params[1:])),
    }


def oos_r2(forecasts, returns, oos_dates):
    """Campbell-Thompson OOS R^2 (%) vs expanding historical mean benchmark."""
    results = {}
    for asset in returns.columns:
        r_actual = returns.loc[oos_dates, asset]
        r_hat    = forecasts.loc[oos_dates, asset]
        r_prev   = returns[asset].expanding().mean().shift(1).loc[oos_dates]
        valid    = r_actual.notna() & r_hat.notna() & r_prev.notna()
        msfe_m   = ((r_actual[valid] - r_hat[valid]) ** 2).mean()
        msfe_b   = ((r_actual[valid] - r_prev[valid]) ** 2).mean()
        results[asset] = (1 - msfe_m / msfe_b) * 100
    return results


def directional_accuracy(forecasts, returns, reg_scores, oos_dates):
    """Fraction of months where sign(forecast) == sign(realized return)."""
    sc_bin  = (reg_scores > 0.5).map({True: "turbulent", False: "calm"})
    results = {}
    for asset in returns.columns:
        r_actual = returns.loc[oos_dates, asset]
        r_hat    = forecasts.loc[oos_dates, asset]
        correct  = (np.sign(r_hat) == np.sign(r_actual))
        for regime in ["calm", "turbulent", "all"]:
            if regime == "all":
                mask = pd.Series(True, index=oos_dates)
            else:
                mask = sc_bin.loc[oos_dates] == regime
            valid = correct.index.intersection(mask[mask].index)
            results[(asset, regime)] = (correct.loc[valid].mean() * 100
                                        if len(valid) > 0 else np.nan)
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  ABLATION  (self-contained, no import from v2)
# ══════════════════════════════════════════════════════════════════════════════

def _build_erc_fixed_lambda(returns, lam=0.94):
    """ERC with fixed EWMA lambda and LW blend -- v2 baseline covariance."""
    asset_names = list(returns.columns)
    oos_dates   = returns.loc[OOS_START:].index
    weights     = pd.DataFrame(index=oos_dates, columns=asset_names, dtype=float)
    cov_stor    = {}
    for t in oos_dates:
        t_loc = returns.index.get_loc(t)
        if t_loc < ROLL_WINDOW:
            continue
        win   = returns.iloc[t_loc - ROLL_WINDOW: t_loc]
        sigma = estimate_sigma(win, lam)
        weights.loc[t]    = erc_weights(sigma)
        cov_stor[str(t)]  = sigma
    return weights.dropna(), cov_stor


def run_ablation(returns_5, returns_7, predictors_7, reg_scores_7):
    """
    Three-step ablation:
      A: 5-asset ERC, fixed lambda=0.94, binary regime, 7 GW predictors
      B: 7-asset ERC (+ INT_BOND + LS_COMMOD), fixed lambda, no overlay
      C: Final -- 7-asset, regime-conditional lambda, continuous score,
         shrinkage, confidence gate, 11 predictors
    """
    def sr(r):  return r.mean()*12 / (r.std()*np.sqrt(12))
    def mdd(r):
        c = (1+r).cumprod(); p = c.cummax(); return ((c-p)/p).min()*100
    def to_pct(w): return w.diff().abs().sum(axis=1).mean()*100

    # ── Step A: 5-asset baseline ──────────────────────────────────────────────
    pred_7gw = predictors_7[["tbl","term_spread","credit_spread","dy","infl","ntis","svar"]]
    pred_5 = pred_7gw.reindex(returns_5.index)

    w_erc_A, cov_A = _build_erc_fixed_lambda(returns_5)

    # Binary regime from FCI on 7-predictor set (same FCI, consistent)
    fci_A       = compute_fci(pred_5)
    exp_med_A   = fci_A.shift(1).expanding(min_periods=REGIME_WARMUP).median()
    bin_reg_A   = pd.Series(index=fci_A.index, dtype=object)
    bin_reg_A[fci_A >  exp_med_A] = "turbulent"
    bin_reg_A[fci_A <= exp_med_A] = "calm"
    # Encode as continuous 0/1 for build_final_weights
    sc_A = (bin_reg_A == "turbulent").astype(float).reindex(w_erc_A.index).fillna(0.5)

    preds_A  = build_predictors_by_asset(returns_5, pred_5)
    oos_A    = returns_5.loc[OOS_START:].index
    fc_A     = regime_predict(returns_5, preds_A, sc_A, oos_A)
    oos_ca   = w_erc_A.index.intersection(fc_A.index)
    w_fa     = build_final_weights(w_erc_A, cov_A, fc_A, sc_A.reindex(w_erc_A.index).rename("score").to_frame()["score"],
                                   shrinkage=1.0, threshold=1.0)
    w_fa     = w_fa.loc[oos_ca]
    r_A      = (returns_5.loc[oos_ca] * w_fa).sum(axis=1)

    # ── Step B: 7-asset ERC, fixed lambda, no overlay ────────────────────────
    w_erc_B, _ = _build_erc_fixed_lambda(returns_7)
    oos_cb = w_erc_B.index.intersection(returns_7.index)
    r_B    = (returns_7.loc[oos_cb] * w_erc_B.loc[oos_cb]).sum(axis=1)

    # ── Step C: final ─────────────────────────────────────────────────────────
    w_erc_C, cov_C = build_erc_weights(returns_7, reg_scores_7)
    preds_C        = build_predictors_by_asset(returns_7, predictors_7)
    oos_all        = returns_7.loc[OOS_START:].index
    fc_C           = regime_predict(returns_7, preds_C, reg_scores_7, oos_all)
    oos_cc         = w_erc_C.index.intersection(fc_C.index).intersection(reg_scores_7.dropna().index)
    w_fc           = build_final_weights(w_erc_C, cov_C, fc_C, reg_scores_7)
    w_fc           = w_fc.loc[oos_cc]
    r_C            = (returns_7.loc[oos_cc] * w_fc).sum(axis=1)

    rows = [
        {"Step": "A: 5-asset, fixed lambda, binary regime, 7 pred",
         "Sharpe": round(sr(r_A), 3), "MDD (%)": round(mdd(r_A), 1),
         "Avg TO (%)": round(to_pct(w_fa), 2)},
        {"Step": "B: +INT_BOND +LS_COMMOD, fixed lambda, ERC only",
         "Sharpe": round(sr(r_B), 3), "MDD (%)": round(mdd(r_B), 1),
         "Avg TO (%)": round(to_pct(w_erc_B.loc[oos_cb]), 2)},
        {"Step": "C: Final (cont. regime, reg-cond lambda, shrink, 11 pred)",
         "Sharpe": round(sr(r_C), 3), "MDD (%)": round(mdd(r_C), 1),
         "Avg TO (%)": round(to_pct(w_fc), 2)},
    ]
    return pd.DataFrame(rows), r_C, w_fc, w_erc_C, cov_C


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def _nber_shading(ax):
    nber = [("1990-07","1991-03"),("2001-03","2001-11"),
            ("2007-12","2009-06"),("2020-02","2020-04")]
    for s, e in nber:
        ax.axvspan(pd.Period(s,"M").to_timestamp(),
                   pd.Period(e,"M").to_timestamp(),
                   color="gray", alpha=0.12, lw=0)


def make_figures(portfolios, weights_dict, reg_scores, factors, oos_common):
    styles = {
        "ERC + Tactical (ours)": {"color": "#1f4e79", "lw": 2.0},
        "ERC base (7 assets)":   {"color": "#2e7d32", "lw": 1.4, "ls": "--"},
        "60/40":                 {"color": "#c62828", "lw": 1.4},
        "Vol-Targeted 60/40":    {"color": "#e65100", "lw": 1.4, "ls": (0,(3,1))},
    }

    # Fig 1: Log cumulative excess returns
    fig, ax = plt.subplots(figsize=(11, 5))
    _nber_shading(ax)
    for name, props in styles.items():
        cum = np.log((1 + portfolios[name]).cumprod())
        ax.plot(cum.index.to_timestamp(), cum.values, label=name, **props)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Log cumulative excess returns (OOS, Jul 1995 - Dec 2024)")
    ax.set_ylabel("Log cumulative excess return")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); fig.savefig("fig1_cumulative.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig 2: Drawdowns
    fig, ax = plt.subplots(figsize=(11, 4.5))
    _nber_shading(ax)
    for name in ["ERC + Tactical (ours)","ERC base (7 assets)","60/40","Equal-weighted (1/N)"]:
        dd = drawdown_series(portfolios[name]) * 100
        ax.plot(dd.index.to_timestamp(), dd.values, label=name)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Drawdowns (OOS)")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); fig.savefig("fig2_drawdowns.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig 3: Rolling 36-month Sharpe
    fig, ax = plt.subplots(figsize=(11, 4.5))
    _nber_shading(ax)
    for name in ["ERC + Tactical (ours)","ERC base (7 assets)","60/40"]:
        rs = rolling_sharpe(portfolios[name], window=36)
        ax.plot(rs.index.to_timestamp(), rs.values, label=name)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Rolling 36-month Sharpe ratio")
    ax.set_ylabel("Sharpe ratio")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); fig.savefig("fig3_rolling_sharpe.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig 4: Weight heatmap
    w_plot = weights_dict["ERC + Tactical (ours)"].T
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(w_plot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.4)
    ax.set_yticks(range(len(w_plot.index))); ax.set_yticklabels(w_plot.index)
    year_starts = [(i, p.year) for i, p in enumerate(w_plot.columns) if p.month == 1]
    ax.set_xticks([i for i,_ in year_starts[::3]])
    ax.set_xticklabels([str(y) for _,y in year_starts[::3]])
    ax.set_title("Weight evolution: ERC + Tactical strategy")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("Weight", fontsize=9)
    fig.tight_layout(); fig.savefig("fig4_weights_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig 5: Rolling 60-month alpha
    ra_f = rolling_alpha_series(portfolios["ERC + Tactical (ours)"], factors, window=60)
    ra_e = rolling_alpha_series(portfolios["ERC base (7 assets)"],   factors, window=60)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    _nber_shading(ax)
    ax.plot(ra_f.index.to_timestamp(), ra_f.values, label="ERC + Tactical (ours)", color="#1f4e79")
    ax.plot(ra_e.index.to_timestamp(), ra_e.values, label="ERC base (7 assets)",   color="#2e7d32", ls="--")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Rolling 60-month annualized alpha (8-factor model)")
    ax.set_ylabel("Alpha (%, annualized)")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); fig.savefig("fig5_rolling_alpha.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig 6: Continuous regime score
    sc_plot = reg_scores.loc[OOS_START:]
    fig, ax = plt.subplots(figsize=(11, 3))
    _nber_shading(ax)
    ax.fill_between(sc_plot.index.to_timestamp(), sc_plot.values, 0.5,
                    where=sc_plot.values > 0.5, alpha=0.4, color="#c62828", label="Turbulent")
    ax.fill_between(sc_plot.index.to_timestamp(), sc_plot.values, 0.5,
                    where=sc_plot.values <= 0.5, alpha=0.4, color="#2e7d32", label="Calm")
    ax.axhline(0.5, color="black", lw=0.8, ls="--")
    ax.set_title("Continuous regime score (FCI expanding percentile rank)")
    ax.set_ylabel("Score  (0 = calmest, 1 = most turbulent)")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); fig.savefig("fig6_regime_score.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Figures saved: fig1 - fig6")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Multi-Asset ERC + Tactical  --  Final (v3)")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    us_eq, rf = load_us_equity_and_rf()
    intl_eq   = load_intl_equity()
    long_treas, credit, pred_gw = load_goyal_welch()
    commod, ls_commod, backw    = load_commodities_and_ls(rf)
    int_bond  = load_int_bond(rf)
    macro     = load_macro_predictors()

    # 7-asset return matrix
    returns = pd.concat([us_eq, intl_eq, long_treas, credit, commod,
                         int_bond, ls_commod], axis=1).loc[START:END]

    # 11 predictors (7 GW + 3 macro + backwardation), backw shifted 1m
    macro2      = macro.reindex(pred_gw.index)
    backw_pred  = backw.shift(1).rename("backwardation")
    predictors  = pd.concat([pred_gw, macro2, backw_pred], axis=1).loc[START:END]

    valid      = (returns.dropna(how="any").index
                  .intersection(predictors.dropna(how="any").index))
    returns    = returns.loc[valid]
    predictors = predictors.loc[valid]

    print(f"  Sample : {returns.index.min()} to {returns.index.max()}  (N={len(returns)})")
    print(f"  Assets : {list(returns.columns)}")
    print(f"  Preds  : {list(predictors.columns)}")

    # 5-asset universe for ablation step A
    returns_5 = returns.drop(columns=["INT_BOND","LS_COMMOD"])

    # ── Regime scores ──────────────────────────────────────────────────────────
    print("\n[2/5] Computing continuous regime scores (expanding FCI percentile)...")
    reg_scores = regime_score_series(predictors)
    reg_scores = reg_scores.reindex(returns.index)
    print(f"  Mean score: {reg_scores.dropna().mean():.3f}  "
          f"Std: {reg_scores.dropna().std():.3f}")

    # ── ERC weights ────────────────────────────────────────────────────────────
    print("\n[3/5] Building ERC weights (regime-conditional EWMA + LW blend)...")
    weights_erc, cov_storage = build_erc_weights(returns, reg_scores)
    print(f"  ERC avg weights:\n{weights_erc.mean().round(3).to_string()}")

    # ── Forecasts ──────────────────────────────────────────────────────────────
    print("\n[4/5] Building regime-conditional combination forecasts (11 predictors + TSMOM)...")
    preds_by  = build_predictors_by_asset(returns, predictors)
    oos_idx   = returns.loc[OOS_START:].index
    forecasts = regime_predict(returns, preds_by, reg_scores, oos_idx)
    print(f"  Forecast shape: {forecasts.shape}")

    # ── Final weights ──────────────────────────────────────────────────────────
    print("\n[5/5] Building final weights (shrink + gate + cont-regime scale)...")
    oos_common = (weights_erc.index
                  .intersection(forecasts.index)
                  .intersection(reg_scores.dropna().index))
    w_final    = build_final_weights(weights_erc, cov_storage, forecasts, reg_scores)
    w_final    = w_final.loc[oos_common]

    R_oos   = returns.loc[oos_common]
    r_final = (w_final * R_oos).sum(axis=1)
    r_erc   = (weights_erc.loc[oos_common] * R_oos).sum(axis=1)
    r_6040  = 0.6 * R_oos["US_EQ"] + 0.4 * R_oos["LONG_TREAS"]
    vt_sc   = ((0.10/np.sqrt(12)) / r_6040.rolling(12).std().shift(1)).clip(upper=1.5).fillna(1.0)
    r_vt    = vt_sc * r_6040
    r_eq    = R_oos.mean(axis=1)

    portfolios = {
        "ERC + Tactical (ours)": r_final,
        "ERC base (7 assets)":   r_erc,
        "60/40":                 r_6040,
        "Vol-Targeted 60/40":    r_vt,
        "Equal-weighted (1/N)":  r_eq,
    }
    weights_dict = {
        "ERC + Tactical (ours)": w_final,
        "ERC base (7 assets)":   weights_erc.loc[oos_common],
    }

    factors = load_factors()

    print(f"\n  OOS period: {oos_common.min()} to {oos_common.max()}  (N={len(oos_common)})")

    # ── Table 1: Headline performance ──────────────────────────────────────────
    print("\n" + "="*60)
    print("TABLE 1: Annualized OOS performance")
    print("="*60)
    rows = []
    for name, r in portfolios.items():
        mu, vol, sr = ann_stats(r)
        mdd_val = max_drawdown(r)
        cal     = mu / abs(mdd_val) if mdd_val != 0 else np.nan
        res     = alpha_regression(r, factors)
        rows.append([name, mu, vol, sr, mdd_val, cal,
                     res["alpha_ann_pct"], res["alpha_t"], res["r2"]])
    df_t1 = pd.DataFrame(rows, columns=[
        "Portfolio","Mean (%)","Vol (%)","Sharpe","MaxDD (%)","Calmar",
        "Alpha (%)","Alpha t","R2 (%)"])
    print(df_t1.round(3).to_string(index=False))

    # ── Table 2: Information ratios ────────────────────────────────────────────
    print("\nTABLE 2: Information ratios vs benchmarks")
    rows = []
    for bn in ["ERC base (7 assets)","60/40","Vol-Targeted 60/40","Equal-weighted (1/N)"]:
        ir, op, te = info_ratio(r_final, portfolios[bn])
        rows.append([bn, op, te, ir])
    df_t2 = pd.DataFrame(rows, columns=["Benchmark","Outperf (%)","TE (%)","IR"])
    print(df_t2.round(3).to_string(index=False))

    # ── Table 3: Factor loadings ───────────────────────────────────────────────
    print("\nTABLE 3: 8-factor alpha and loadings (Newey-West, 6 lags)")
    rows = []
    for name in ["ERC + Tactical (ours)","ERC base (7 assets)","60/40","Equal-weighted (1/N)"]:
        res = alpha_regression(portfolios[name], factors)
        rows.append([name, res["alpha_ann_pct"], res["alpha_t"], res["r2"]])
    df_t3 = pd.DataFrame(rows, columns=["Portfolio","Alpha (%)","t-stat","R2 (%)"])
    print(df_t3.round(3).to_string(index=False))
    res_f = alpha_regression(r_final, factors)
    print("\n  Factor betas (ERC + Tactical):")
    for k, v in res_f["betas"].items():
        print(f"    {k:8s}: {v:.3f}")

    # ── Table 4: Regime-conditional returns ────────────────────────────────────
    print("\nTABLE 4: Returns by regime (calm vs turbulent)")
    sc_bin = (reg_scores > 0.5).map({True: "turbulent", False: "calm"})
    sc_oos = sc_bin.loc[oos_common]
    rows = []
    for name, r in portfolios.items():
        rc = r[sc_oos == "calm"]
        rt = r[sc_oos == "turbulent"]
        t_stat, _ = scipy_stats.ttest_ind(rc.dropna(), rt.dropna(), equal_var=False)
        rows.append([name, rc.mean()*12*100, rt.mean()*12*100,
                     (rc.mean()-rt.mean())*12*100, round(t_stat, 2)])
    df_t4 = pd.DataFrame(rows, columns=["Portfolio","Calm (%)","Turb (%)","Diff (%)","t-stat"])
    print(df_t4.round(3).to_string(index=False))

    # ── Table 5: Turnover and net-of-cost ─────────────────────────────────────
    print("\nTABLE 5: Turnover and net-of-cost Sharpe (@10 bps)")
    rows = []
    for name, w in weights_dict.items():
        to_avg  = avg_turnover_pct(w)
        r_net   = portfolios[name] - monthly_turnover(w).reindex(
            portfolios[name].index, fill_value=0) * (COST_BPS / 1e4)
        mu, vol, sr = ann_stats(r_net)
        res_net = alpha_regression(r_net, factors)
        rows.append([name, to_avg, mu, sr, res_net["alpha_ann_pct"]])
    df_t5 = pd.DataFrame(rows, columns=["Portfolio","Avg TO (%)","Net Mean (%)","Net Sharpe","Net Alpha (%)"])
    print(df_t5.round(3).to_string(index=False))

    # ── Table 6: TC sensitivity ────────────────────────────────────────────────
    print("\nTABLE 6: Transaction-cost sensitivity")
    to_f = monthly_turnover(w_final)
    to_e = monthly_turnover(weights_erc.loc[oos_common])
    rows = []
    for bps in [0, 5, 10, 15, 20, 25, 50, 100]:
        rn = r_final - to_f.reindex(r_final.index, fill_value=0) * (bps / 1e4)
        re = r_erc   - to_e.reindex(r_erc.index,   fill_value=0) * (bps / 1e4)
        _, _, sr_n = ann_stats(rn)
        ir, _, _   = info_ratio(rn, re)
        rows.append([bps, round(sr_n, 3), round(ir, 3)])
    df_t6 = pd.DataFrame(rows, columns=["Cost (bps)","Net Sharpe","Net IR vs ERC base"])
    print(df_t6.to_string(index=False))

    # ── Table 7: Ablation ─────────────────────────────────────────────────────
    print("\nTABLE 7: Extension ablation")
    df_t7, _, _, _, _ = run_ablation(returns_5, returns, predictors, reg_scores)
    print(df_t7.to_string(index=False))

    # ── Table 8: Directional accuracy ─────────────────────────────────────────
    print("\nTABLE 8: Directional accuracy (%)")
    da = directional_accuracy(forecasts, returns, reg_scores, oos_common)
    rows = []
    for asset in returns.columns:
        rows.append({"Asset": asset,
                     "All":       round(da.get((asset,"all"),np.nan), 1),
                     "Calm":      round(da.get((asset,"calm"),np.nan), 1),
                     "Turbulent": round(da.get((asset,"turbulent"),np.nan), 1)})
    df_t8 = pd.DataFrame(rows)
    print(df_t8.to_string(index=False))

    # ── Table 9: OOS R2 ───────────────────────────────────────────────────────
    print("\nTABLE 9: OOS R2 (%) -- Campbell-Thompson vs expanding mean")
    r2v = oos_r2(forecasts, returns, oos_common)
    df_t9 = pd.DataFrame([{"Asset": a, "OOS R2 (%)": round(r2v[a], 3)}
                           for a in returns.columns])
    print(df_t9.to_string(index=False))

    # ── Table 10: Net Calmar ──────────────────────────────────────────────────
    print("\nTABLE 10: Net Calmar (@10 bps)")
    rows = []
    for name, r in portfolios.items():
        if name in weights_dict:
            rn = r - monthly_turnover(weights_dict[name]).reindex(
                r.index, fill_value=0) * (COST_BPS / 1e4)
        else:
            rn = r
        mu, _, _ = ann_stats(rn)
        mdd_n    = max_drawdown(rn)
        rows.append([name, round(mu,3), round(mdd_n,2), round(mu/abs(mdd_n),3)])
    df_t10 = pd.DataFrame(rows, columns=["Portfolio","Net Mean (%)","Net MaxDD (%)","Net Calmar"])
    print(df_t10.to_string(index=False))

    # ── Save tables and figures ────────────────────────────────────────────────
    for df, name in [(df_t1,"table1_performance"),(df_t2,"table2_ir"),
                     (df_t3,"table3_alpha"),(df_t4,"table4_regime"),
                     (df_t5,"table5_turnover"),(df_t6,"table6_tc_sensitivity"),
                     (df_t7,"table7_ablation"),(df_t8,"table8_directional"),
                     (df_t9,"table9_oos_r2"),(df_t10,"table10_net_calmar")]:
        df.to_csv(f"{name}.csv", index=False)

    print("\nBuilding figures...")
    make_figures(portfolios, weights_dict, reg_scores, factors, oos_common)

    # ── Pickle for reuse ──────────────────────────────────────────────────────
    pd.to_pickle(portfolios,   "portfolio_returns.pkl")
    pd.to_pickle(weights_dict, "portfolio_weights.pkl")
    pd.to_pickle(reg_scores,   "regime_scores.pkl")

    print(f"\nERC solver fallback count: {_erc_fallback_count}")
    print("\nDone. Summary:")
    main_row = [r for r in rows if "ERC + Tactical" in r[0]][0] if rows else None
    res_main = alpha_regression(r_final, factors)
    mu_f, vol_f, sr_f = ann_stats(r_final)
    print(f"  Strategy Sharpe : {sr_f:.3f}  (60/40: 0.724)")
    print(f"  Strategy MDD    : {max_drawdown(r_final):.1f}%  (60/40: -30.7%)")
    print(f"  Alpha           : {res_main['alpha_ann_pct']:.3f}%  t={res_main['alpha_t']:.2f}")
    print(f"  R2              : {res_main['r2']:.1f}%")


if __name__ == "__main__":
    main()
