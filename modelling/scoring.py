from data_prep.team_history import History
import numpy as np


class Scoring:
    def __init__(self, mapping, proxy):
        self.mapping = mapping
        self.proxy = proxy

    def score_data(self, games):
        mapping = self.mapping
        proxy = self.proxy

        scoring = []
        for i in games:
            home = i[0]
            away = i[1]
            home_df = History(mapping, proxy).team_roll(home, season=0, shift=0, web=True).tail(1)[
                ['Rnd', 'F_mean', 'F_std', 'A_mean', 'A_std', 'M_mean', 'A_std', 'R_mean', 'perc']]
            home_df['Rnd'] = home_df['Rnd'] + 1
            away_df = History(mapping, proxy).team_roll(away, season=0, shift=0, web=True).tail(1)[
                ['F_mean', 'F_std', 'A_mean', 'A_std', 'M_mean', 'A_std', 'R_mean', 'perc']]
            features = np.concatenate([home_df.values[0], away_df.values[0]], axis=0)
            scoring.append(features)
        return scoring
