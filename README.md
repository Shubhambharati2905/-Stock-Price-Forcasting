# AAPL Stock Price Forecasting
### A Hybrid ARIMA + LSTM Framework

This project forecasts the daily closing price of **(AAPL)** using a two-stage hybrid model:

1. **ARIMA** captures the *linear* structure of the price series through a rolling walk-forward forecast.
2. **LSTM** learns the *non-linear* structure left behind in the ARIMA residuals.

The two components are combined into a single hybrid forecast, which is then evaluated with standard error metrics *and* a directional trading backtest — the core theme of this project is that low error metrics alone do **not** prove a model is useful.

> **Note on figures:** All numbers below reflect one specific run. The notebook pulls a live rolling 10-year window via `yfinance`, so re-running on a different day will shift the results marginally.

---

## 1. Objective

- Forecast the next-day closing price of AAPL using a hybrid ARIMA + LSTM model.
- ARIMA captures the linear, autocorrelated structure of the price series via rolling walk-forward forecasting.
- LSTM learns any non-linear structure remaining in the ARIMA residuals.
- Combine both into a single hybrid forecast (ARIMA output + LSTM residual correction).
- Benchmark the hybrid against a **naive persistence baseline** (tomorrow = today) to check if it beats "no change."
- Validate real-world usefulness via **directional accuracy** and a **long/short trading backtest**, not just RMSE/MAPE.
- **Core goal:** test whether the hybrid delivers a genuine predictive edge — not just a low error number that looks good on a near-random-walk series.

---

## 2. Methodology & Workflow

### 2.1 Setup & Data Acquisition

Download 10 years of AAPL history and split it chronologically into an 80% training window and a 20% test window (no shuffling, no leakage).

### 2.2 Stationarity & Seasonal Decomposition

ARIMA requires a stationary series. An Augmented Dickey-Fuller (ADF) test confirms that a single first-difference (`d = 1`) is enough, and the training series is decomposed into trend, seasonal, and residual components on an approximate annual (252 trading-day) cycle.

<img width="1389" height="1015" alt="Unknown-6" src="https://github.com/user-attachments/assets/b66b60a1-f4ef-446c-a8bd-51b1e8048faf" />

### 2.3 ACF / PACF — reading candidate `p` and `q`

On the stationary (differenced) series, the ACF suggests the moving-average order `q` and the PACF suggests the autoregressive order `p`.
<img width="1590" height="490" alt="Unknown" src="https://github.com/user-attachments/assets/2177a125-0f48-471a-a0e7-b71807cbf032" />

 and PACF plots" src="https://github.com/user-attachments/assets/Unknown" />

### 2.4 ARIMA Order Selection (AIC grid search)

Rather than fixing `ARIMA(1, 1, 1)` by hand, a small grid search over `p, q ∈ {0..3}` with `d = 1` is run once on the training set. `ARIMA(1,1,1)` is used as the final order: it converges cleanly and sits within ~12 AIC points of the (non-converging) top-scoring candidate.

### 2.5 Rolling Walk-Forward ARIMA

Forecast one step at a time across the test window. After each step the *actual* observed price is fed back into the model so it always sees the most recent history. For speed the model is fit **once** and extended with `append(..., refit=False)` instead of refitting from scratch every day.

**ARIMA diagnostics** — a zoomed view of the standalone ARIMA forecast, plus the in-sample and out-of-sample residuals it leaves behind (these residuals are exactly what the LSTM tries to model).

<img width="1489" height="590" alt="Unknown-7" src="https://github.com/user-attachments/assets/87d87c8e-a685-408c-9d4e-7389d52f62cc" />


### 2.6 LSTM on the ARIMA Residuals

The LSTM learns non-linear patterns remaining in the ARIMA residuals. Residuals are scaled to `[-1, 1]` (the scaler is fit on **training residuals only** to avoid leakage), then reshaped into overlapping 10-day lookback windows.

### 2.7 Hybrid Forecast & Evaluation

The final forecast is `ARIMA (linear) + LSTM (non-linear correction)`, scored on RMSE, MAE, and MAPE on the held-out test window and benchmarked against the naive baseline.

<img width="1489" height="590" alt="Unknown-3" src="https://github.com/user-attachments/assets/8ceb21a3-3028-4029-850b-2d4bfb1ba3b9" />

### 2.8 Reality Check — Directional Accuracy & Trading Backtest

Low price error does not guarantee a useful model. This step measures how often the model predicts the correct **direction** (up vs. down relative to the previous day) and simulates a simple long/short strategy: go long when the model predicts an up day, short when it predicts a down day.

<img width="1489" height="590" alt="Equity curve backtest" src="https://github.com/user-attachments/<img width="1489" height="590" alt="Unknown-5" src="https://github.com/user-attachments/assets/3b131acf-f123-4d36-a55a-0d917d1168e6" />
assets/Unknown-5" />

---

## 3. Empirical Observations & Performance Metrics

Out-of-sample error over the **503 trading days** in the test window, benchmarked against a naive "tomorrow = today" baseline:

| Model | RMSE | MAE | MAPE |
|---|---|---|---|
| **Naive (persistence)** | **4.089** | **2.817** | **1.187** |
| ARIMA only | 4.097 | 2.822 | 1.190% |
| ARIMA + LSTM hybrid | 4.098 | 2.809 | 1.185% |

> **Hybrid vs. Naive baseline: −0.22% RMSE** — the hybrid is actually *slightly worse* than simply predicting no change.

Directional and trading results:

| Evaluation Category | Value / Performance Metric |
|---|---|
| Total Trading Days Evaluated | 503 Days |
| Correct Directional Predictions | 271 Days |
| True Directional Win Rate | 53.88% |
| Buy & Hold AAPL Return | 42.89% |
| Hybrid ARIMA+LSTM Strategy Return | 14.89% |

- **The Accuracy Illusion:** A 1.19% MAPE looks near-perfect, but the naive baseline beats it (1.402%). On a near-random-walk series, low error just reflects the day-to-day autocorrelation of price *levels* — not genuine predictive skill.
- **The LSTM's Real Contribution:** The residual-learning stage does not improve on the naive baseline at all, and its validation loss plateaus from epoch 1 — evidence that little non-linear signal remains in the ARIMA residuals to extract.
- **Directional Reality Check:** A 53.68% win rate is only marginally better than a coin flip, and it is unstable — an earlier model configuration on the same data produced a 51.7% win rate and a *negative* strategy return, underscoring how sensitive these headline numbers are to small model changes.

---

## 4. Quantitative Inferences

### The Profit vs. Win-Rate Paradox

A **53.88%** directional win rate is barely above a coin flip — daily AAPL movement remains largely unpredictable. Yet the strategy still returned **43.76%**, well below Buy & Hold's **95.41%**. This isn't skill, it's **diluted exposure**: roughly half the days are spent short against the prevailing trend, so the strategy rides much less of the uptrend than simply holding the stock.

This is also unstable — an earlier ARIMA configuration on the same data produced a **51.69%** win rate and a **negative** return of **−1.43%**. A small model change flipped a gain into a loss, which isn't how a durable edge behaves.

### Real-World Trading Friction

The backtest assumes zero friction. Flipping positions on most of 503 test days would incur real costs:

- Brokerage commissions
- Bid-ask spread and slippage
- Exchange/regulatory fees

Even **0.05% per trade** across hundreds of trades would eat into the already modest 43.76% return — and given how close the win rate is to 50%, it wouldn't take much to push this negative again.

---

## 5. Conclusion

The hybrid ARIMA + LSTM framework offers a structured, multi-layered approach to price tracking. While the walk-forward loop produces low error metrics (RMSE/MAPE), these numbers are misleading on their own — a naive "no change" baseline scores *slightly better*, and true validation requires a directional backtest. Here, that backtest reveals a win rate barely above chance and returns well below simple Buy & Hold, showing that low error does not translate into real trading skill once market noise and trading friction are accounted for. The real deliverable is a correctly engineered, leak-free pipeline and the diagnostics that expose *why* daily-price hybrid models struggle to beat a random walk.

---
