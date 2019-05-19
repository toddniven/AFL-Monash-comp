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
                ['Rnd', 'F_mean', 'F_std', 'A_mean', 'A_std', 'M_mean', 'A_std', 'W_sum', 'perc']]
            home_df['Rnd'] = home_df['Rnd'] + 1
            away_df = History(mapping, proxy).team_roll(away, season=0, shift=0, web=True).tail(1)[
                ['F_mean', 'F_std', 'A_mean', 'A_std', 'M_mean', 'A_std', 'W_sum', 'perc']]
            features1 = np.concatenate([home_df.values[0], away_df.values[0]], axis=0)

            home_df2 = \
            History(mapping, proxy).team_roll_ha(home, home_away='H', season=0, shift=0, web=True).tail(1)[
                ['F_mean', 'F_std', 'A_mean', 'A_std', 'M_mean', 'A_std', 'W_sum', 'perc']]
            away_df2 = \
            History(mapping, proxy).team_roll_ha(home, home_away='A', season=0, shift=0, web=True).tail(1)[
                ['F_mean', 'F_std', 'A_mean', 'A_std', 'M_mean', 'A_std', 'W_sum', 'perc']]
            features2 = np.concatenate([home_df2.values[0], away_df2.values[0]], axis=0)
            scoring.append(np.concatenate([features1, features2], axis=0))
        return scoring
