import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score

df = pd.read_csv('ipl_matches_with_L5_stats.csv')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['year'] = df['date'].dt.year

home_cities = {
    'Chennai Super Kings': 'Chennai',
    'Delhi Capitals': 'Delhi',
    'Gujarat Titans': 'Ahmedabad',
    'Kolkata Knight Riders': 'Kolkata',
    'Lucknow Super Giants': 'Lucknow',
    'Mumbai Indians': 'Mumbai',
    'Punjab Kings': 'Chandigarh',
    'Rajasthan Royals': 'Jaipur',
    'Royal Challengers Bengaluru': 'Bengaluru',
    'Sunrisers Hyderabad': 'Hyderabad',
    'Deccan Chargers': 'Hyderabad'
}

top = ['Bengaluru', 'Chennai', 'Mumbai', 'Kolkata', 'Delhi', 'Hyderabad', 'Chandigarh', 'Jaipur', 'Ahmedabad', 'Lucknow']

past = df[df['winner'].notna() & (df['winner'] != 'Draw/No Result') & (df['year'] <= 2024)].copy()
fut = df[df['year'] == 2025].copy()
past['target'] = (past['winner'] == past['team1']).astype(int)

for d in [past, fut]:
    d['team1_bf'] = (((d['toss_decision'] == 'bat') & (d['toss_winner'] == d['team1'])) |
                     ((d['toss_decision'] == 'field') & (d['toss_winner'] == d['team2']))).astype(int)
    d['elo_diff'] = d['team1_elo'] - d['team2_elo']
    d['city'] = d['city'].fillna('Unknown').replace({'Bangalore': 'Bengaluru'})

comb = pd.concat([past, fut])
comb['city_c'] = comb['city'].apply(lambda x: x if x in top else 'Other')
dm = pd.get_dummies(comb['city_c'], prefix='city', dtype=int)
fin = pd.concat([comb, dm], axis=1)

fin['form_diff'] = fin['team1_l5_sr'] - fin['team2_l5_sr']
fin['eco_diff'] = fin['team1_l5_ec'] - fin['team2_l5_ec']
fin['h2h_diff'] = fin['team1_h2h_adv'] - fin['team2_h2h_adv']
fin['team1_is_home'] = (fin['team1'].map(home_cities) == fin['city']).astype(int)
fin['team2_is_home'] = (fin['team2'].map(home_cities) == fin['city']).astype(int)
fin['chase_wr_diff'] = fin['team1_chase_wr'] - fin['team2_chase_wr']
fin['defend_wr_diff'] = fin['team1_defend_wr'] - fin['team2_defend_wr']
fin['pp_sr_diff'] = fin['team1_pp_sr'] - fin['team2_pp_sr']
fin['death_ec_diff'] = fin['team1_death_ec'] - fin['team2_death_ec']
fin['h2h_wr_diff'] = fin['team1_h2h_wr'] - fin['team2_h2h_wr']
fin['recent_wr_diff'] = fin['team1_recent_wr'] - fin['team2_recent_wr']
fin = fin.fillna(0)

city_cols = [c for c in dm.columns if c in fin.columns]

features = [
    'team1_elo', 'team2_elo', 'elo_diff',
    'team1_bf', 'is_night_match',
    'team1_l5_sr', 'team2_l5_sr', 'form_diff',
    'team1_l5_ec', 'team2_l5_ec', 'eco_diff',
    'team1_chase_wr', 'team2_chase_wr', 'chase_wr_diff',
    'team1_defend_wr', 'team2_defend_wr', 'defend_wr_diff',
    'team1_star_power', 'team2_star_power',
    'team1_in_form_count', 'team2_in_form_count',
    'h2h_diff', 'h2h_wr_diff',
    'team1_is_home', 'team2_is_home',
    'team1_pp_sr', 'team2_pp_sr', 'pp_sr_diff',
    'team1_death_ec', 'team2_death_ec', 'death_ec_diff',
    'team1_recent_wr', 'team2_recent_wr', 'recent_wr_diff',
    'avg_first_inn_score', 'avg_second_inn_score', 'avg_pp_score', 'venue_chase_wr'
] + city_cols

train = fin[fin['year'] <= 2024].copy().reset_index(drop=True)
test = fin[fin['year'] == 2025].copy()

current_year = 2025
train['weight'] = train['year'].apply(lambda y: 0.5 ** ((current_year - y) / 4))

test = test[test['winner'].notna() & (test['winner'] != 'Draw/No Result')].copy()
test['target'] = (test['winner'] == test['team1']).astype(int)

param_grid = {
    'n_estimators': [200, 250, 300],
    'learning_rate': [0.005, 0.01, 0.02, 0.1],
    'max_depth': [3, 4, 5, 6],
    'gamma': [0.1, 0.3, 0.5, 0.7, 1],
    'subsample': [0.8, 0.85, 0.9],
    'colsample_bytree': [0.75, 0.80, 0.85],
    'min_child_weight': [1, 3, 5],
}

tscv = TimeSeriesSplit(n_splits=5)
base_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')

search = RandomizedSearchCV(
    base_model,
    param_grid,
    n_iter=200,
    cv=tscv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

search.fit(train[features], train['target'], sample_weight=train['weight'])

print(f"\nBest params: {search.best_params_}")
print(f"Best CV accuracy: {search.best_score_ * 100:.2f}%")

best_model = search.best_estimator_
test_pred = (best_model.predict_proba(test[features])[:, 1] > 0.5).astype(int)
acc = accuracy_score(test['target'], test_pred)
print(f"2025 Test accuracy with best params: {acc * 100:.2f}%")

print("\nCopy these into train_model.py:")
print(search.best_params_)