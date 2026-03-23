import pandas as pd

matches = pd.read_csv('ipl_matches_with_L5_stats.csv')
deliveries = pd.read_csv('ipl_deliveries_flat.csv')

matches['match_id'] = matches['match_id'].astype(int)
deliveries['match_id'] = deliveries['match_id'].astype(int)
matches['date'] = pd.to_datetime(matches['date'], errors='coerce')

venue_stats = {}

for venue, grp in matches.groupby('venue'):
    mids = grp['match_id'].tolist()
    match_dels = deliveries[deliveries['match_id'].isin(mids)]

    first_inn = match_dels[match_dels['inning'] == 1]
    second_inn = match_dels[match_dels['inning'] == 2]

    avg_first_inn_score = first_inn.groupby('match_id')['total_runs'].sum().mean()
    avg_second_inn_score = second_inn.groupby('match_id')['total_runs'].sum().mean()

    pp_dels = match_dels[match_dels['over'] <= 5]
    avg_pp_score = pp_dels.groupby('match_id')['total_runs'].sum().mean()

    valid = grp[grp['winner'].notna() & (grp['winner'] != 'Draw/No Result')]
    chasing_wins = 0
    total = 0

    for _, row in valid.iterrows():
        t1_bf = (((row['toss_decision'] == 'bat') & (row['toss_winner'] == row['team1'])) |
                ((row['toss_decision'] == 'field') & (row['toss_winner'] == row['team2'])))
        chasing_team = row['team2'] if t1_bf else row['team1']
        if row['winner'] == chasing_team:
            chasing_wins += 1
        total += 1

    wicket_dels = match_dels[match_dels['is_wicket'] == 1].copy()
    wicket_dels['bowler_type'] = 'unknown'
    pace_bowlers = ['Anderson', 'Broad', 'Bumrah', 'Shami', 'Starc', 'Hazlewood', 'Morkel', 'Southee', 'Kumar', 'Cummins', 'Rabada', 'Boult', 'Trent', 'Woakes', 'Archer']
    total_wickets = len(wicket_dels)

    venue_stats[venue] = {
        'avg_first_inn_score': round(avg_first_inn_score , 2) if not pd.isna(avg_first_inn_score) else 160,
        'avg_second_inn_score': round(avg_second_inn_score, 2) if not pd.isna(avg_second_inn_score) else 150,
        'avg_pp_score': round(avg_pp_score, 2) if not pd.isna(avg_pp_score) else 48,
        'venue_chase_wr' : round(chasing_wins / total, 4) if total > 0 else 0.5,
        'total_matches': total
    }

venue_df = pd.DataFrame.from_dict(venue_stats, orient='index').reset_index().rename(columns={'index': 'venue'})
venue_df.columns = ['venue', 'avg_first_inn_score', 'avg_second_inn_score', 'avg_pp_score', 'venue_chase_wr', 'total_matches']
venue_df = venue_df[venue_df['total_matches'] >= 5].copy()
venue_df.to_csv('venue_stats.csv', index=False)

print(venue_df.sort_values('total_matches', ascending=False).head(20))
