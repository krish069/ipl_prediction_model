import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score

df = pd.read_csv('ipl_matches_with_L5_stats.csv')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['year'] = df['date'].dt.year

past = df[df['winner'].notna() & (df['winner'] != 'Draw/No Result') & (df['year'] <= 2024)].copy()
fut = df[df['year'] == 2025].copy()
past['target'] = (past['winner'] == past['team1']).astype(int)

home_cities = {
    'Chennai Super Kings': 'Chennai',
    'Delhi Capitals': 'Delhi',
    'Delhi Daredevils': 'Delhi',
    'Gujarat Titans': 'Ahmedabad',
    'Kolkata Knight Riders': 'Kolkata',
    'Lucknow Super Giants': 'Lucknow',
    'Mumbai Indians': 'Mumbai',
    'Punjab Kings': 'Chandigarh',
    'Kings XI Punjab': 'Chandigarh',
    'Rajasthan Royals': 'Jaipur',
    'Royal Challengers Bengaluru': 'Bengaluru',
    'Royal Challengers Bangalore': 'Bengaluru',
    'Sunrisers Hyderabad': 'Hyderabad',
    'Deccan Chargers': 'Hyderabad'
}

top = ['Bengaluru', 'Chennai', 'Mumbai', 'Kolkata', 'Delhi', 'Hyderabad', 'Chandigarh', 'Jaipur', 'Ahmedabad', 'Lucknow']

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
    'team1_bf',
    'team1_l5_sr', 'team2_l5_sr', 'form_diff',
    'team1_l5_ec', 'team2_l5_ec', 'eco_diff',
    'team1_chase_wr', 'team2_chase_wr',
    'team1_star_power', 'team2_star_power',
    'team1_in_form_count', 'team2_in_form_count',
    'h2h_diff',
    'team1_is_home', 'team2_is_home',
    'team1_pp_sr', 'team2_pp_sr',       
    'team1_death_ec', 'team2_death_ec'
]


train = fin[fin['year'] <= 2024].copy()
test = fin[fin['year'] == 2025].copy()

current_year = 2025
train['weight'] = train['year'].apply(lambda y: 0.5 ** ((current_year - y) / 4))

has_test_labels = not test['winner'].isna().all()
if has_test_labels:
    test = test[test['winner'].notna() & (test['winner'] != 'Draw/No Result')].copy()
    test['target'] = (test['winner'] == test['team1']).astype(int)

model = xgb.XGBClassifier(
    n_estimators = 250,
    learning_rate = 0.01,
    max_depth = 4,
    subsample = 0.85,
    colsample_bytree = 0.85,
    min_child_weight = 1,
    gamma = 1,
    random_state = 42
)

model.fit(train[features], train['target'], sample_weight=train['weight'])

test_prob = model.predict_proba(test[features])[:, 1]
test_pred = (test_prob > 0.50).astype(int)

res = test.copy()
res['predicted_winner'] = [row['team1'] if p == 1 else row['team2'] for p, row in zip(test_pred, res.to_dict('records'))]
res['confidence'] = [max(p) for p in model.predict_proba(test[features])]
res.to_csv('2025_predictions.csv', index=False)

if has_test_labels:
    acc = accuracy_score(test['target'], test_pred)
    print(f"2025 LIVE ACCURACY: {acc * 100:.2f}%\n")

importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance)
model.save_model('ipl_xgboost_model.json')


results = []
for seed in range(10):
    model = xgb.XGBClassifier(
        n_estimators=250, learning_rate=0.01, max_depth=4,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=1,
        gamma=1, random_state=seed
    )
    model.fit(train[features], train['target'])
    pred = (model.predict_proba(test[features])[:, 1] > 0.5).astype(int)
    results.append(accuracy_score(test['target'], pred) * 100)

print(f"Mean: {sum(results)/len(results):.2f}%")
print(f"Min: {min(results):.2f}%  Max: {max(results):.2f}%")
print(results)