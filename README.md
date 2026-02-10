# MarketLens

MarketLens is a **regime-aware, probabilistic market analysis system** designed for  
**uncertainty-aware diagnostics and decision support**.

It is **not** a trading bot and **not** a price prediction engine.  
Instead of producing confident or actionable signals, MarketLens focuses on
**understanding market conditions, uncertainty, and when signals are unreliable**.

---

## Motivation

Financial markets do not behave the same way all the time.

A small price movement during a calm market can mean something very different from
the same movement during a volatile or chaotic period.  
Most forecasting systems ignore this and assume a single, stable market behavior.

MarketLens does not.

It explicitly accounts for **changing market conditions** and communicates
**uncertainty instead of false confidence**.

---

## RAPID Framework

MarketLens implements the **RAPID framework**:

**RAPID — Regime-Aware Probabilistic Inference for Dynamics**

The framework is built on four core ideas:

- Markets operate under different **regimes** (calm, unstable, volatile)
- Forecasts should be **probabilistic**, not single-number predictions
- Uncertainty should be **explicit and visible**
- When data is unreliable, the system should **refuse to make strong claims**

RAPID emphasizes knowing **when not to trust a signal** as much as identifying
when a weak signal exists.

---

## What MarketLens Does

MarketLens provides a **5-day market outlook** with an emphasis on
context and uncertainty rather than precision.

Specifically, it:

- Forecasts **5-day forward log returns** using probabilistic outputs
- Produces **uncertainty bands** (P10 / P50 / P90) instead of point estimates
- Estimates **directional confidence** only when statistically justified
- Identifies **market regimes** based on volatility structure
- Automatically **rejects assets** with insufficient or imbalanced data
- Performs stability checks during extreme market conditions

The system is intentionally conservative and avoids overconfident conclusions.

---

## What MarketLens Does NOT Do

- ❌ No buy/sell recommendations  
- ❌ No trading or execution logic  
- ❌ No guaranteed profitability  
- ❌ No real-time decision making  

MarketLens is a **diagnostic and interpretive tool**, not an automated trading system.

---

## Model Architecture (High-Level)

MarketLens uses a **deep temporal model** designed for numerical time-series data:

- **CNN layers** capture short-term temporal dependencies
- **Transformer attention** models longer-range context
- A shared representation feeds multiple outputs:
  - Probabilistic return forecast (P10 / P50 / P90)
  - Direction confidence (Up / Down / Neutral)
  - Market regime classification (Low / Medium / High volatility)

The architecture prioritizes **stability, interpretability, and uncertainty awareness**
over aggressive prediction.

---

## Evaluation & Guardrails

MarketLens applies strict validation and safety checks:

- Regression error is measured on **unscaled log returns**
- Directional AUC is reported **only when statistically valid**
- Single-class or unreliable evaluations return **NaN**, not fabricated scores
- Assets with insufficient data or extreme imbalance are **automatically rejected**

These guardrails prevent misleading results and enforce honest evaluation.

---

## Dashboard Overview

The Streamlit dashboard presents results in a **layman-friendly, non-misleading way**:

- A probabilistic 5-day outlook
- Forecast uncertainty visualization
- Market condition (regime) context
- Directional confidence indicators
- Stability and residual diagnostics
- Stress-test behavior during extreme market events

The dashboard is designed for **interpretation and understanding**, not action.

---

## Stress Testing

MarketLens includes stress-test analysis on historical crisis periods
(e.g., the COVID-19 market crash).

The goal is **not** to predict crashes, but to verify that:

- Predictions remain bounded
- Uncertainty increases appropriately
- The model does not behave erratically under extreme conditions

This helps assess model **robustness and reliability**.

---

## Intended Use

MarketLens is suitable for:

- Educational exploration of financial time-series modeling
- Research on regime-aware and probabilistic forecasting
- Diagnostic analysis of market behavior
- Demonstrating uncertainty-aware ML system design

It is **not intended** for live trading or financial decision automation.

---

## Setup & Usage

MarketLens is designed to be easy to run locally for **analysis and exploration**.

### Requirements

- Python 3.9+
- CPU-only environment (GPU optional)

Install dependencies:

```bash
pip install -r requirements.txt
```

### Running Experiments

To train and evaluate the system on a small set of assets:

```bash
python run_experiment.py --train_top 5
```

This will:
- Run baseline models
- Select statistically valid assets
- Train the deep model only where data quality allows
- Save metrics and artifacts for inspection

To run the pipeline for a single asset:

```bash
python run_experiment.py --ticker <TICKER_SYMBOL>
```

### Launching the Dashboard

Start the Streamlit dashboard with:

```bash
streamlit run app.py
```

The dashboard provides:
- A 5-day probabilistic outlook
- Forecast uncertainty visualization
- Market condition (regime) context
- Stability and residual diagnostics

### Notes

Large datasets, trained models, and generated reports are intentionally excluded
from the repository.

The system may reject certain assets automatically if data quality or class balance
is insufficient. This behavior is expected and part of the system’s design.

---

## Disclaimer

This project is for **educational and research purposes only**.  
It does not constitute financial or investment advice.

---

## Author

**Ayush Ranjan**  
