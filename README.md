# CRASH-Gambling-risk-engine

A Machine Learning pipeline designed to detect high-risk, addictive betting behavior on Crash gambling platforms.This engine models  psychological patterns to flag accounts exhibiting severe loss-chasing behavior.

---

##  Business & Compliance Impact
* **Target Audience:** Trust, Safety, and Responsible Gaming Compliance Teams.
* **Core Philosophy:** **Human-in-the-Loop (HITL) Detection.** In gambling risk mitigation, a False Negative (missing an addict ) carries severe ethical and regulatory consequences, whereas a False Positive (reviewing a normal account) carries low operational cost.
* **Objective:** Maximize **Class 1 Recall** to ensure high-risk user trajectories are caught early for human compliance review.

---

## Engineered Behavioral Features
Raw transactional logs are aggregated per user into key psychological metrics:

* **`loss_chase_ratio`:** Average percentage increase in wager size immediately following a loss (quantifies emotional "tilt").
* **`wager_volatility`:** Ratio of standard deviation to mean bet size (identifies erratic bankroll management).
* **`scalper_ratio`:** Proportion of total bets placed on ultra-low multipliers ($\le 1.10\text{x}$) to exploit safety nets.

---

##  Model Iterations & Performance

The pipeline was evaluated across two distinct structural paradigms to address **Proxy Leakage**:

| Iteration | Features Included | Target Class Recall | Target Class Precision | Key Takeaway |
| :--- | :--- | :--- | :--- | :--- |
| **Model 1 (Volume Included)** | Volume + Ratios | **93%** | **49%** | High recall, but heavily anchored on `total_wagered` (acted as a VIP big-spender detector). |
| **Model 2 (Pure Psychological)** | Ratios Only | **64%** | **15%** | Successfully isolated behavioral pathology from capital size. |

> **Strategic Architecture:** To ensure zero reliance on bankroll size while maintaining high sensitivity, the operational deployment utilizes low decision thresholds on pure psychological features to feed an internal compliance queue.

---
##  Quickstart Guide

### 1. Clone Repository & Install Dependencies

git clone [https://github.com/vshalovr9000/CRASH-Gambling-risk-engine.git](https://github.com/vshalovr9000/CRASH-Gambling-risk-engine.git)
cd CRASH-Gambling-risk-engine
pip install -r requirements.txt


### 2. Run feature pipeline module
python src/feature_pipeline.py

### 3. Launch Streamlit Web App
streamlit run app.py

