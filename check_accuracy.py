import pandas as pd

pred = pd.read_csv('2025_predictions.csv')
actual = pd.read_csv('ipl_matches_with_L5_stats.csv')[['date', 'team1', 'team2', 'winner']]

merged = pd.merge(pred, actual, on=['date', 'team1', 'team2'], how='inner')

merged = merged.dropna(subset=['winner_y'])
merged = merged[merged['winner_y'] != 'Draw/No Result']

print((merged['predicted_winner'] == merged['winner_y']).mean() * 100)