\# IPL Match Predictor 🏏



An automated machine learning pipeline that predicts Indian Premier League (IPL) match outcomes using XGBoost.



Instead of just looking at basic win/loss records, this model builds custom features for every match, including:

\* \*\*Dynamic Elo Ratings:\*\* Calculates real-time team strength based on historical match data.

\* \*\*Recent Form:\*\* Tracks rolling stats for the last 5 matches, including player strike rates, death over economy, and team momentum.

\* \*\*Match Context:\*\* Factors in toss decisions, home-ground advantage, and day/night conditions.



\## 📊 Model Performance



To ensure realistic, out-of-sample validation, the model was trained on historical IPL data from \*\*2008 to 2024\*\* and tested strictly on unseen \*\*2025 match data\*\*. 



\* \*\*Mean Accuracy:\*\* 70.0%

\* \*\*Accuracy Range:\*\* 68.5% — 72.8%

\* \*\*Algorithm:\*\* `XGBClassifier`

\* \*\*Top Drivers of Win Probability:\*\* Team Elo Differential, Toss Decision (Bat/Field First advantage), and Recent Team Strike Rates.



\## 🚀 Quick Start



\*\*1. Clone the repository\*\*

```bash

git clone \[https://github.com/krish069/ipl\_prediction\_model.git](https://github.com/krish069/ipl\_prediction\_model.git)

cd ipl\_prediction\_model

