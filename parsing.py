import os
import json
import pandas as pd

def parse_json(file_path):
    match_metadata = []
    delivery_data = []

    name_mapping = {
        'Delhi Daredevils': 'Delhi Capitals',
        'Kings XI Punjab': 'Punjab Kings',
        'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
        'Rising Pune Supergiant': 'Rising Pune Supergiants',
        'Deccan Chargers': 'Sunrisers Hyderabad'
    }

    for filename in os.listdir(file_path):
        if filename.endswith('.json'):
            with open(os.path.join(file_path, filename), 'r') as f:
                data = json.load(f)
                match_id = filename.split('.')[0]

                info = data.get('info', {})

                match_meta = {
                    'match_id': match_id,
                    'season': info.get('season', 'Unknown'),
                    'date': info.get('dates', ['Unknown'])[0],
                    'city': info.get('city', 'Unknown'),
                    'venue': info.get('venue', 'Unknown'),
                    'team1': name_mapping.get(info.get('teams', [None, None])[0], info.get('teams', [None, None])[0]),
                    'team2': name_mapping.get(info.get('teams', [None, None])[1], info.get('teams', [None, None])[1]),
                    'toss_winner': name_mapping.get(info.get('toss', {}).get('winner', 'Unknown'), info.get('toss', {}).get('winner', 'Unknown')),
                    'toss_decision': info.get('toss', {}).get('decision', 'Unknown'),
                    'winner': name_mapping.get(info.get('outcome', {}).get('winner', 'Draw/No Result'), info.get('outcome', {}).get('winner', 'Draw/No Result')),
                    'player_of_the_match': info.get('player_of_match', [None])[0]
                }

                match_metadata.append(match_meta)

                for inning_idx, inning in enumerate(data.get('innings', [])):
                    team_batting = inning.get('team')

                    for over in inning.get('overs', []):
                        over_num = over.get('over')

                        for ball_num, ball in enumerate(over.get('deliveries', [])):
                            runs = ball.get('runs', {})
                            extras_detail = ball.get('extras', {})
                            is_wicket = 1 if 'wickets' in ball else 0

                            delivery_meta = {
                                'match_id': match_id,
                                'inning': inning_idx + 1,
                                'batting_team': team_batting,
                                'over': over_num,
                                'ball_in_over': ball_num + 1,
                                'batter': ball.get('batter'),
                                'bowler': ball.get('bowler'),
                                'non_striker': ball.get('non_striker'),
                                'batter_runs': runs.get('batter', 0),
                                'extra_runs': runs.get('extras', 0),
                                'total_runs': runs.get('total', 0),
                                'wide_runs': extras_detail.get('wides', 0),
                                'noball_runs': extras_detail.get('noballs', 0),
                                'is_wicket': is_wicket
                            }

                            delivery_data.append(delivery_meta)

    matches_df = pd.DataFrame(match_metadata)
    deliveries_df = pd.DataFrame(delivery_data)

    matches_df['date'] = pd.to_datetime(matches_df['date'], errors='coerce')
    matches_df = matches_df.sort_values('date').reset_index(drop=True)

    return matches_df, deliveries_df

matches, deliveries = parse_json('./ipl_data')

matches.to_csv('ipl_matches_flat.csv', index=False)
deliveries.to_csv('ipl_deliveries_flat.csv', index=False)