import os
import pandas as pd
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

if not os.path.exists('ipl_xgboost_model.json'):
    print("Missing File: 'ipl_xgboost_model.json' was not found.")
    print("Please run your 'train_model.py' script to generate and save the model file before running this predictor.")
    exit()

try:
    df = pd.read_csv('ipl_matches_with_L5_stats.csv')
    model = xgb.XGBClassifier()
    model.load_model('ipl_xgboost_model.json')
except Exception as e:
    print("Error loading model or data:", e)
    exit()

def get_latest_stats(team_name, data):
    team_data = data[(data['team1'] == team_name) | (data['team2'] == team_name)].copy()
    if team_data.empty:
        return None
    
    latest_match = team_data.sort_values('date', ascending=False).iloc[0]
    is_team1 = (latest_match['team1'] == team_name)

    pre_match_elo = latest_match['team1_elo'] if is_team1 else latest_match['team2_elo']
    opp_elo = latest_match['team2_elo'] if is_team1 else latest_match['team1_elo']
    expected_win = 1 / (1 + 10 ** ((opp_elo - pre_match_elo) / 400))
    actual_win = 1 if latest_match['winner'] == team_name else 0
    post_match_elo = pre_match_elo + 20 * (actual_win - expected_win) 

    return {
        'elo' : post_match_elo,
        'l5_sr': latest_match['team1_l5_sr'] if is_team1 else latest_match['team2_l5_sr'],
        'l5_ec': latest_match['team1_l5_ec'] if is_team1 else latest_match['team2_l5_ec'],
        'l5_pp_sr': latest_match['team1_pp_sr'] if is_team1 else latest_match['team2_pp_sr'],
        'l5_death_ec': latest_match['team1_death_ec'] if is_team1 else latest_match['team2_death_ec'],
        'chase_wr': latest_match['team1_chase_wr'] if is_team1 else latest_match['team2_chase_wr'],
        'defend_wr': latest_match['team1_defend_wr'] if is_team1 else latest_match['team2_defend_wr'],
        'star_power': latest_match['team1_star_power'] if is_team1 else latest_match['team2_star_power'],
        'in_form': latest_match['team1_in_form_count'] if is_team1 else latest_match['team2_in_form_count']
    }

teams = sorted(list(set(df['team1'].unique()) | set(df['team2'].unique())))

print("\nAvailable teams:")
for i, team in enumerate(teams):
    print(f"[{i}] {team}")

try:
    team1_idx = int(input("\nEnter Team 1 index: "))
    team2_idx = int(input("Enter Team 2 index: "))

    t1_name = teams[team1_idx]
    t2_name = teams[team2_idx]

    toss_winner = int(input(f"Who won the toss? (0 for {t1_name}, 1 for {t2_name}): "))
    toss_decision = int(input("Toss decision? (0 for Bat, 1 for Field): "))
    
    is_night = int(input("Is this a Night Match? (1 for Yes, 0 for Day): "))
    
    top_cities = ['Bengaluru', 'Chennai', 'Mumbai', 'Kolkata', 'Delhi', 'Hyderabad', 'Chandigarh', 'Jaipur', 'Ahmedabad', 'Lucknow']
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
    display_cities = top_cities + ['Other']
    
    print("\nAvailable Cities:")
    for i, city in enumerate(display_cities):
        print(f"[{i}] {city}")
        
    city_idx = int(input("\nEnter City index: "))
    city_name = display_cities[city_idx]

except Exception:
    print("Invalid Input. Terminating.")
    exit()

t1_stats = get_latest_stats(t1_name, df)
t2_stats = get_latest_stats(t2_name, df)

toss_winner_name = t1_name if toss_winner == 0 else t2_name

sorted_teams = sorted([t1_name, t2_name])
t1_name, t2_name = sorted_teams[0], sorted_teams[1]
t1_stats = get_latest_stats(t1_name, df)
t2_stats = get_latest_stats(t2_name, df)

team1_bf = int((toss_decision == 0 and toss_winner_name == t1_name) or (toss_decision == 1 and toss_winner_name == t2_name))

input_data = {
    'team1_elo': t1_stats['elo'],
    'team2_elo': t2_stats['elo'],
    'elo_diff': t1_stats['elo'] - t2_stats['elo'],
    'team1_bf': team1_bf,
    'is_night_match': is_night,
    'team1_l5_sr': t1_stats['l5_sr'],
    'team2_l5_sr': t2_stats['l5_sr'],
    'form_diff': t1_stats['l5_sr'] - t2_stats['l5_sr'],
    'team1_l5_ec': t1_stats['l5_ec'],
    'team2_l5_ec': t2_stats['l5_ec'],
    'eco_diff': t1_stats['l5_ec'] - t2_stats['l5_ec'],
    'team1_chase_wr': t1_stats['chase_wr'],
    'team2_chase_wr': t2_stats['chase_wr'],
    'team1_defend_wr': t1_stats['defend_wr'],
    'team2_defend_wr': t2_stats['defend_wr'],
    'defend_wr_diff': t1_stats['defend_wr'] - t2_stats['defend_wr'],
    'team1_pp_sr': t1_stats['l5_pp_sr'],
    'team2_pp_sr': t2_stats['l5_pp_sr'],
    'team1_death_ec': t1_stats['l5_death_ec'],
    'team2_death_ec': t2_stats['l5_death_ec'],
    'team1_star_power': t1_stats['star_power'],
    'team2_star_power': t2_stats['star_power'],
    'team1_in_form_count': t1_stats['in_form'],
    'team2_in_form_count': t2_stats['in_form'],
    'h2h_diff': 0,
    'team1_is_home': 1 if home_cities.get(t1_name) == city_name else 0,
    'team2_is_home': 1 if home_cities.get(t2_name) == city_name else 0,
}

for c in top_cities + ['Other']:
    input_data[f'city_{c}'] = 1 if city_name == c else 0

live_df = pd.DataFrame([input_data])
model_features = model.get_booster().feature_names
live_df = live_df[model_features]

probabilities = model.predict_proba(live_df)[0]
t1_prob = probabilities[1] * 100
t2_prob = probabilities[0] * 100

if t1_prob > t2_prob:
    print(f"\nPredicted Winner: {t1_name} with {t1_prob:.2f}% confidence")
else:
    print(f"\nPredicted Winner: {t2_name} with {t2_prob:.2f}% confidence")

print("\nFeature Importances:")
importance = pd.DataFrame({
    'feature': model_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(importance)