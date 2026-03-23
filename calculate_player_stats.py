import pandas as pd
from collections import deque
import warnings
warnings.filterwarnings('ignore')

matches_df = pd.read_csv('ipl_matches_with_elo.csv')
deliveries_df = pd.read_csv('ipl_deliveries_flat.csv')
venue_df = pd.read_csv('venue_stats.csv')

matches_df['match_id'] = matches_df['match_id'].astype(str)
deliveries_df['match_id'] = deliveries_df['match_id'].astype(str)

matches_df['date'] = pd.to_datetime(matches_df['date'], errors='coerce')
matches_df = matches_df.sort_values(['date', 'match_id']).reset_index(drop=True)
matches_df['is_weekend'] = matches_df['date'].dt.dayofweek.isin([5, 6])
matches_df['match_rank_on_day'] = matches_df.groupby('date').cumcount()
matches_df['is_night_match'] = (
    (~matches_df['is_weekend']) |  
    (matches_df['match_rank_on_day'] > 0) 
).astype(int)
matches_df.drop(columns=['is_weekend', 'match_rank_on_day'], inplace=True)

season_first_innings = {}
for season, grp in matches_df.groupby('season'):
    mids = grp['match_id'].tolist()
    first_inn = deliveries_df[(deliveries_df['match_id'].isin(mids)) & (deliveries_df['inning'] == 1)]
    median_score = first_inn.groupby('match_id')['total_runs'].sum().median()
    season_first_innings[season] = median_score if not pd.isna(median_score) else 175

player_batting_history = {}
player_bowling_history = {}
player_mom_counts = {}
team_situational_history = {}
h2h_dismissals = {}
h2h_match_wins = {}
player_pp_batting = {}
player_death_bowling = {}
team_recent_wins = {}

team1_l5_strike_rates, team2_l5_strike_rates = [], []
team1_l5_economy_rates, team2_l5_economy_rates = [], []
team1_chase_win_rates, team2_chase_win_rates = [], []
team1_defend_win_rates, team2_defend_win_rates = [], []
team1_star_power_index, team2_star_power_index = [], []
team1_in_form_counts, team2_in_form_counts = [], []
team1_h2h_adv_list, team2_h2h_adv_list = [], []
team1_l5_pp_sr, team2_l5_pp_sr = [], []
team1_l5_death_ec, team2_l5_death_ec = [], []
team1_h2h_win_rates, team2_h2h_win_rates = [], []
team1_recent_win_rates, team2_recent_win_rates = [], []

def get_rolling_sr(players):
    runs = sum(sum(inn[0] for inn in player_batting_history.get(p, [])) for p in players)
    balls = sum(sum(inn[1] for inn in player_batting_history.get(p, [])) for p in players)
    return (runs / balls * 100) if balls > 0 else 120.0

def get_rolling_economy(players):
    runs_conceded = sum(sum(inn[0] for inn in player_bowling_history.get(p, [])) for p in players)
    balls_bowled = sum(sum(inn[1] for inn in player_bowling_history.get(p, [])) for p in players)
    return (runs_conceded / (balls_bowled / 6)) if balls_bowled > 0 else 8.5

def get_pp_sr(players):
    runs = sum(sum(inn[0] for inn in player_pp_batting.get(p, [])) for p in players)
    balls = sum(sum(inn[1] for inn in player_pp_batting.get(p, [])) for p in players)
    return (runs / balls * 100) if balls > 0 else 120.0

def get_death_ec(players):
    runs = sum(sum(inn[0] for inn in player_death_bowling.get(p, [])) for p in players)
    balls = sum(sum(inn[1] for inn in player_death_bowling.get(p, [])) for p in players)
    return (runs / (balls / 6)) if balls > 0 else 9.5

def count_in_form(players):
    return sum(1 for p in players if sum(1 for inn in player_batting_history.get(p, []) if inn[0] >= 30) >= 3)

def get_win_rate(team, mode):
    s = team_situational_history.get(team, {'cw': 0, 'ct': 0, 'dw': 0, 'dt': 0})
    if mode == 'chase':
        return s['cw'] / s['ct'] if s['ct'] > 0 else 0.5
    return s['dw'] / s['dt'] if s['dt'] > 0 else 0.5

def get_h2h_win_rate(t1, t2):
    pair = tuple(sorted([t1, t2]))
    history = list(h2h_match_wins.get(pair, []))
    if not history:
        return 0.5
    rate = sum(history) / len(history)
    return rate if t1 == pair[0] else 1 - rate

def get_recent_win_rate(team):
    history = list(team_recent_wins.get(team, []))
    if not history:
        return 0.5
    return sum(history) / len(history)

for index, row in matches_df.iterrows():
    m_id, t1, t2 = row['match_id'], row['team1'], row['team2']
    season = row['season']
    match_deliveries = deliveries_df[deliveries_df['match_id'] == m_id]

    t1_batters = match_deliveries[match_deliveries['batting_team'] == t1]['batter'].unique()
    t2_batters = match_deliveries[match_deliveries['batting_team'] == t2]['batter'].unique()
    t1_bowlers = match_deliveries[match_deliveries['batting_team'] == t2]['bowler'].unique()
    t2_bowlers = match_deliveries[match_deliveries['batting_team'] == t1]['bowler'].unique()

    t1_h2h_advantage = sum(h2h_dismissals.get((b, bw), 0) for b in t2_batters for bw in t1_bowlers)
    t2_h2h_advantage = sum(h2h_dismissals.get((b, bw), 0) for b in t1_batters for bw in t2_bowlers)
    team1_h2h_adv_list.append(t1_h2h_advantage)
    team2_h2h_adv_list.append(t2_h2h_advantage)

    team1_h2h_win_rates.append(get_h2h_win_rate(t1, t2))
    team2_h2h_win_rates.append(get_h2h_win_rate(t2, t1))
    team1_recent_win_rates.append(get_recent_win_rate(t1))
    team2_recent_win_rates.append(get_recent_win_rate(t2))

    team1_l5_strike_rates.append(get_rolling_sr(t1_batters))
    team2_l5_strike_rates.append(get_rolling_sr(t2_batters))
    team1_l5_economy_rates.append(get_rolling_economy(t1_bowlers))
    team2_l5_economy_rates.append(get_rolling_economy(t2_bowlers))
    team1_in_form_counts.append(count_in_form(t1_batters))
    team2_in_form_counts.append(count_in_form(t2_batters))
    team1_star_power_index.append(sum(player_mom_counts.get(p, 0) for p in t1_batters))
    team2_star_power_index.append(sum(player_mom_counts.get(p, 0) for p in t2_batters))

    team1_chase_win_rates.append(get_win_rate(t1, 'chase'))
    team1_defend_win_rates.append(get_win_rate(t1, 'defend'))
    team2_chase_win_rates.append(get_win_rate(t2, 'chase'))
    team2_defend_win_rates.append(get_win_rate(t2, 'defend'))

    team1_l5_pp_sr.append(get_pp_sr(t1_batters))
    team2_l5_pp_sr.append(get_pp_sr(t2_batters))
    team1_l5_death_ec.append(get_death_ec(t1_bowlers))
    team2_l5_death_ec.append(get_death_ec(t2_bowlers))

    match_bat_data = {}
    for _, d in match_deliveries.iterrows():
        p, r, ex = d['batter'], d['batter_runs'], d['extra_runs']
        if d.get('is_wicket', 0) == 1 and pd.notna(d['bowler']):
            pair = (d['batter'], d['bowler'])
            h2h_dismissals[pair] = h2h_dismissals.get(pair, 0) + 1
        if p not in match_bat_data:
            match_bat_data[p] = [0, 0]
        match_bat_data[p][0] += r
        if not (ex > 0 and r == 0):
            match_bat_data[p][1] += 1
    for p, stats in match_bat_data.items():
        if p not in player_batting_history:
            player_batting_history[p] = deque(maxlen=5)
        player_batting_history[p].append(stats)

    match_bowl_data = {}
    for _, d in match_deliveries.iterrows():
        p, rc, wr, nb = d['bowler'], d['total_runs'], d['wide_runs'], d['noball_runs']
        if p not in match_bowl_data:
            match_bowl_data[p] = [0, 0]
        match_bowl_data[p][0] += rc
        if not (wr > 0 or nb > 0):
            match_bowl_data[p][1] += 1
    for p, stats in match_bowl_data.items():
        if p not in player_bowling_history:
            player_bowling_history[p] = deque(maxlen=5)
        player_bowling_history[p].append(stats)

    for p in t1_batters.tolist() + t2_batters.tolist():
        p_deliveries = match_deliveries[(match_deliveries['batter'] == p) & (match_deliveries['over'] <= 5)]
        if not p_deliveries.empty:
            runs = p_deliveries['batter_runs'].sum()
            balls = len(p_deliveries[~((p_deliveries['extra_runs'] > 0) & (p_deliveries['batter_runs'] == 0))])
            if p not in player_pp_batting:
                player_pp_batting[p] = deque(maxlen=5)
            player_pp_batting[p].append([runs, balls])

    for p in t1_bowlers.tolist() + t2_bowlers.tolist():
        p_deliveries = match_deliveries[(match_deliveries['bowler'] == p) & (match_deliveries['over'] >= 16)]
        if not p_deliveries.empty:
            runs = p_deliveries['total_runs'].sum()
            balls = len(p_deliveries[~((p_deliveries['wide_runs'] > 0) | (p_deliveries['noball_runs'] > 0))])
            if p not in player_death_bowling:
                player_death_bowling[p] = deque(maxlen=5)
            player_death_bowling[p].append([runs, balls])

    winner, mom = row['winner'], row.get('player_of_the_match')
    if pd.notna(mom):
        player_mom_counts[mom] = player_mom_counts.get(mom, 0) + 1

    if pd.notna(winner) and winner != 'Draw/No Result':
        t1_bf = (row['toss_winner'] == t1 and row['toss_decision'] == 'bat') or \
                (row['toss_winner'] != t1 and row['toss_decision'] == 'field')
        batting_first_team = t1 if t1_bf else t2
        innings_total = match_deliveries[match_deliveries['batting_team'] == batting_first_team]['total_runs'].sum()
        threshold = season_first_innings.get(season, 170)

        for t in [t1, t2]:
            if t not in team_situational_history:
                team_situational_history[t] = {'cw': 0, 'ct': 0, 'dw': 0, 'dt': 0}
            is_win = winner == t
            is_bf = (t == t1 and t1_bf) or (t == t2 and not t1_bf)
            if not is_bf and innings_total > threshold:
                team_situational_history[t]['ct'] += 1
                if is_win:
                    team_situational_history[t]['cw'] += 1
            elif is_bf and innings_total > threshold:
                team_situational_history[t]['dt'] += 1
                if is_win:
                    team_situational_history[t]['dw'] += 1

        pair = tuple(sorted([t1, t2]))
        if pair not in h2h_match_wins:
            h2h_match_wins[pair] = deque(maxlen=5)
        h2h_match_wins[pair].append(1 if winner == pair[0] else 0)

        for t in [t1, t2]:
            if t not in team_recent_wins:
                team_recent_wins[t] = deque(maxlen=5)
            team_recent_wins[t].append(1 if winner == t else 0)

matches_df['team1_l5_sr'], matches_df['team2_l5_sr'] = team1_l5_strike_rates, team2_l5_strike_rates
matches_df['team1_l5_ec'], matches_df['team2_l5_ec'] = team1_l5_economy_rates, team2_l5_economy_rates
matches_df['team1_chase_wr'], matches_df['team2_chase_wr'] = team1_chase_win_rates, team2_chase_win_rates
matches_df['team1_defend_wr'], matches_df['team2_defend_wr'] = team1_defend_win_rates, team2_defend_win_rates
matches_df['team1_star_power'], matches_df['team2_star_power'] = team1_star_power_index, team2_star_power_index
matches_df['team1_in_form_count'], matches_df['team2_in_form_count'] = team1_in_form_counts, team2_in_form_counts
matches_df['team1_h2h_adv'], matches_df['team2_h2h_adv'] = team1_h2h_adv_list, team2_h2h_adv_list
matches_df['team1_pp_sr'], matches_df['team2_pp_sr'] = team1_l5_pp_sr, team2_l5_pp_sr
matches_df['team1_death_ec'], matches_df['team2_death_ec'] = team1_l5_death_ec, team2_l5_death_ec
matches_df['team1_h2h_wr'], matches_df['team2_h2h_wr'] = team1_h2h_win_rates, team2_h2h_win_rates
matches_df['team1_recent_wr'], matches_df['team2_recent_wr'] = team1_recent_win_rates, team2_recent_win_rates
matches_df = matches_df.merge(venue_df[['venue', 'avg_first_inn_score', 'avg_second_inn_score', 'avg_pp_score', 'venue_chase_wr']],
                               on='venue', how='left')
matches_df['avg_first_inn_score'].fillna(160, inplace=True)
matches_df['avg_second_inn_score'].fillna(150, inplace=True)
matches_df['avg_pp_score'].fillna(48, inplace=True)
matches_df['venue_chase_wr'].fillna(0.5, inplace=True)
matches_df.to_csv('ipl_matches_with_L5_stats.csv', index=False)
