import pandas as pd

def calculate_elo(file_path):
    matches = pd.read_csv(file_path)
    matches['date'] = pd.to_datetime(matches['date'], errors='coerce')
    matches = matches.sort_values('date').reset_index(drop=True)

    name_mapping = {
        'Delhi Daredevils': 'Delhi Capitals',
        'Kings XI Punjab': 'Punjab Kings',
        'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
        'Rising Pune Supergiant': 'Rising Pune Supergiants',
        'Deccan Chargers': 'Sunrisers Hyderabad'
    }
    matches['team1'] = matches['team1'].replace(name_mapping)
    matches['team2'] = matches['team2'].replace(name_mapping)
    matches['winner'] = matches['winner'].replace(name_mapping)

    initial_elo = 1500
    k_factor = 32

    team_elo = {}
    current_season = None

    def expected_score(elo_a, elo_b):
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

    team1_prematch_elo = []
    team2_prematch_elo = []

    for _, row in matches.iterrows():
        team1 = row['team1']
        team2 = row['team2']
        winner = row['winner']
        season = row['season']

        if season != current_season:
            current_season = season
            for team in list(team_elo.keys()):
                team_elo[team] = 0.75 * team_elo[team] + 0.25 * initial_elo

        if team1 not in team_elo:
            team_elo[team1] = initial_elo
        if team2 not in team_elo:
            team_elo[team2] = initial_elo

        elo_team1 = team_elo[team1]
        elo_team2 = team_elo[team2]

        team1_prematch_elo.append(elo_team1)
        team2_prematch_elo.append(elo_team2)

        expected_team1 = expected_score(elo_team1, elo_team2)
        expected_team2 = expected_score(elo_team2, elo_team1)

        if winner == team1:
            score_team1, score_team2 = 1, 0
        elif winner == team2:
            score_team1, score_team2 = 0, 1
        else:
            score_team1, score_team2 = 0.5, 0.5

        team_elo[team1] += k_factor * (score_team1 - expected_team1)
        team_elo[team2] += k_factor * (score_team2 - expected_team2)

    matches['team1_elo'] = team1_prematch_elo
    matches['team2_elo'] = team2_prematch_elo
    matches['elo_diff'] = matches['team1_elo'] - matches['team2_elo']

    matches.to_csv('ipl_matches_with_elo.csv', index=False)
    print("Elo ratings calculated and saved to ipl_matches_with_elo.csv")

    print("\nCurrent Elo ratings for teams:")
    for team, elo in sorted(team_elo.items(), key=lambda x: x[1], reverse=True):
        print(f"{team}: {elo:.2f}")

calculate_elo('ipl_matches_flat.csv')