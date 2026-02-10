# 📈 Regime-Aware Stock Trend System: A Business Overview

## 🎯 The Philosophy: Don't Predict Price, Predict *Advantage*
Most AI models fail in finance because they try to predict the exact price tomorrow. This is mathematically impossible due to noise. 

**We take a different approach:**
Instead of guessing the *noise*, we categorize the *signal*.

---

## 💡 Core Innovations

### 1. "Signal First" Architecture (Hybrid CNN-Transformer)
We combine two specialized neural networks to find edge:
*   **The Scanner (CNN)**: Like a technical trader, it spots immediate patterns—volatility shifts, momentum breaks, and sudden volume spikes.
*   **The Analyst (Transformer)**: Like a macro strategist, it remembers 6 months of context to decide if the current move is a trend or a trap.

### 2. The "Wait for Signal" Gate (AUC > 0.55)
A bad model is worse than no model. Our system is rigorous:
*   **Automatic Rejection**: If a model cannot beat random chance (AUC > 0.55) or a simple Moving Average rule, it is **automatically discarded**.
*   **Result**: We only deploy models that have statistically significant predictive power.

### 3. Direction Over Magnitude
We focus on the 5-Day Trend (Up vs Down).
*   **Why?** Predicting *how much* a stock will move is hard. Predicting *where* it is going is easier and more profitable.
*   **Neutral Zone**: 
    *   **Training**: We ignore small moves (noise) so the model learns only from significant trends.
    *   **Trading**: If the model's confidence is low (e.g., 52% prob), we stay in **Cash** (Neutral). We only trade when the signal is strong.

### 4. Profitability Validation (Backtesting)
Metrics like "Loss" are for machines. "Sharpe Ratio" is for humans.
*   We interpret our models through a **Backtest Simulator** that trades the Test Set.
*   This proves whether the AI's "accuracy" actually translates to **Profit & Loss (PnL)**.

---

## ⚠️ Important Disclaimer
*   **Not Financial Advice**: This system is a research tool. Real-world trading involves costs (slippage, fees, spread) not fully modeled here.
*   **Success Metrics**: A "Successful" model has a consistent **AUC > 0.55** across multiple tickers and positive backtest returns. Note that market regimes change, and past performance is not a guarantee of future results.

---

## 🛡️ Selection Policy: Quality over Quantity

1.  **Mass Screening**: We scan thousands of tickers with lightweight algorithms to find candidates (Baseline RMSE).
2.  **Deep Training**: We train complex Hybrid Models on the top 50 candidates.
3.  **The Acceptance Gate**: We ONLY keep the model if it proves itself:
    *   **RMSE check**: Beat the Rolling Mean?
    *   **AUC check**: Beat random chance (> 0.55)?
    *   If not, it is discarded.

---

## 💻 Hardware Agnostic
This system is built to run anywhere, democratizing high-end financial ML:
*   **NVIDIA GPU**: Full CUDA acceleration for fastest training.
*   **Intel/AMD AI PCs**: Leverages **NPU** (Neural Processing Unit) via DirectML.
*   **Standard Laptop**: Optimized to run inference on a standard CPU in <100ms.

---

## 🏆 Summary: Why This Matters
| Feature | Old School AI | Our System |
| :--- | :--- | :--- |
| **Goal** | Guess Price ($100.52) | **Capture Trend (Up/Down)** |
| **Validation** | "Low Error" | **"High Sharpe Ratio"** |
| **Context** | Blind to Volatility | **Adapts to Market Regimes** |
| **Reliability** | Hallucinates Patterns | **Rejects Noise (AUC Check)** |

*Built for robust, statistically significant edge.*
