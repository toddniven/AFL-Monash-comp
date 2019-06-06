from data_prep.team_history import History
from data_prep.feature_eng import Features
import numpy as np
import pandas as pd


class Scoring:
    def __init__(self, mapping, proxy):
        self.mapping = mapping
        self.proxy = proxy

    def score_data(self, games):
        """
        Prepare the most recent round for scoring.
        :param games:
        :return:
        """
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
        return  Features().div_cols(scoring)


class Simulate:
    """
    Class to use models from Scoring to simulate scores.
    """
    def __init__(self, mapping, proxy):
        self.mapping = mapping
        self.proxy = proxy

    @staticmethod
    def score_f(y_true, y_pred):
        return 1 + np.log2(y_true * y_pred + (1 - y_true) * (1 - y_pred))

    def generate_past_scores(self, data_path, best_models, team_df):
        """
        Use models to simulate past scores (based on score_f above) and output each as numpy arrays ready to be used
        as features
        :param self:
        :param best_models: Input the season models
        :param team_df:
        :return:
        """
        mapping = self.mapping
        proxy = self.proxy
        for season in range(1, len(best_models)+1):
            X = np.load(data_path + '/training-' + str(2019 - season) + '.npy')
            X_train = Features().div_cols(X).values
            y = np.load(data_path + '/results-' + str(2019 - season) + '.npy')

            score = Simulate.score_f(y, best_models[season - 1].predict_proba(X_train)[:, 1])

            year = str(2019 - season)
            teams = list(mapping.keys())
            teams.remove('Kangaroos')

            if season >= 8:
                teams.remove('Greater Western Sydney')
            if season >= 9:
                teams.remove('Gold Coast')

            out = pd.DataFrame()
            for team in teams:
                df = History(mapping, proxy).team_roll(team, season, team_df)
                home_df = df[df['T'] == 'H'].reset_index(drop=True)
                l = len(home_df)
                out = pd.concat([out, pd.DataFrame(np.c_[[year] * l, [team] * l, home_df['Opponent']])],
                                axis=0,
                                ignore_index=True)
            out.columns = ['year', 'home', 'away']
            out['score'] = score
            out = out.set_index(['year', 'home'])

            arr1 = out["score"].groupby(['year', 'home']).transform(
                lambda x: x.cumsum().shift()).values
            arr2 = out["score"].groupby(['year', 'home']).transform(
                lambda x: x.rolling(20, min_periods=1).std().shift()).values
            arr3 = out["score"].groupby(['year', 'home']).transform(
                lambda x: x.rolling(20, min_periods=1).mean().shift()).values

            np.save(data_path + '/scores-' + str(2019 - season) + '.npy', np.c_[arr1, arr2, arr3])
        return None
