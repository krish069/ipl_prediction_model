import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from xgboost import train
from train_model import model, features, train
import pandas as pd

importance = sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True)[:20]
feats = [i[0] for i in importance]
imps = [i[1] for i in importance]

sns.barplot(x=imps, y=feats)
plt.title('Top 20 Drivers of IPL Wins')
plt.show()

y_pred = model.predict(train[features])
cm = confusion_matrix(train['target'], y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

y_prob = model.predict_proba(train[features])[:, 1]
fpr, tpr, _ = roc_curve(train['target'], y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.legend()
plt.title('ROC Curve')
plt.show()

df = pd.read_csv('ipl_matches_with_elo.csv')
df['date'] = pd.to_datetime(df['date'])

teams = ['Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bengaluru']

plt.figure(figsize=(12, 6))

for team in teams:
    team_matches = df[(df['team1'] == team) | (df['team2'] == team)].copy()
    
    team_matches['elo'] = team_matches.apply(
        lambda row: row['team1_elo'] if row['team1'] == team else row['team2_elo'],
        axis=1
    )
    
    team_matches['elo_smooth'] = team_matches['elo'].rolling(10).mean()
    plt.plot(team_matches.index, team_matches['elo_smooth'], label=team, linewidth=2.5)

plt.title('Elo Rating Over Time')
plt.xlabel('Match Progression')
plt.ylabel('Elo Rating')
plt.legend()
plt.show()