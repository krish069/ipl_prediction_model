# IPL Match Predictor 🏏

An automated machine learning pipeline that predicts Indian Premier League (IPL) match outcomes using XGBoost.

Instead of just looking at basic win/loss records, this model builds custom features for every match, including:
* **Dynamic Elo Ratings:** Calculates real-time team strength based on historical match data.
* **Recent Form:** Tracks rolling stats for the last 5 matches, including player strike rates, death over economy, and team momentum.
* **Match Context:** Factors in toss decisions, home-ground advantage, and day/night conditions.

##  Model Performance

To ensure realistic, out-of-sample validation, the model was trained on historical IPL data from **2008 to 2024** and tested strictly on unseen **2025 match data**. 

* **Mean Accuracy:** 70.0%
* **Accuracy Range:** 68.5% — 72.8%
* **Algorithm:** `XGBClassifier`
* **Top Drivers of Win Probability:** Team Elo Differential, Toss Decision (Bat/Field First advantage), and Recent Team Strike Rates.

##  To Use it
```bash
git clone [https://github.com/krish069/ipl_prediction_model.git](https://github.com/krish069/ipl_prediction_model.git)
cd ipl_prediction_model
python auto_updater.py
python live_predictor.py
```

## Brief Description of Files
```
auto_updater.py - Fetches the latest raw data and runs the entire data pipeline.
parsing.py - Cleans and flattens the raw JSON ball-by-ball data.
calculate_elo.py, calculate_player_stats.py, calculate_venue_stats.py - Scripts that generate features.
train_model.py / tune_model.py - Trains and optimizes the XGBoost classifier.
live_predictor.py - The CLI tool for inputting match conditions and getting a prediction.
```
