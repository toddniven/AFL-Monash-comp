import pandas as pd
import numpy as np
from data_prep.web_scraping import Scrape


class History:
    def __init__(self, mapping, proxy):
        self.mapping = mapping
        self.proxy = proxy

    def generate_team_history(self, season_list=range(1, 16)):
        mapping = self.mapping
        proxy = self.proxy
        teams = list(mapping.keys())
        teams.remove('Kangaroos')
        team_df = {}  # dictionary of all dataframes
        for team in teams:
            for season in season_list:
                print(team, season)
                if (season < 8 or team != 'Greater Western Sydney') and (season < 9 or team != 'Gold Coast'):
                    team_df[team, season] = Scrape(mapping, proxy).scrape_history(team, season)
        return team_df

    def generate_game_data(self, data_path, team_df, season_list=range(1, 16)):
        mapping = self.mapping
        proxy = self.proxy

        for season in season_list:
            year = str(2019 - season)
            print(year)
            teams = list(mapping.keys())
            teams.remove('Kangaroos')

            if season >= 8:
                teams.remove('Greater Western Sydney')
            if season >= 9:
                teams.remove('Gold Coast')

            team_hg = []
            results = []

            for team in teams:
                df = History(mapping, proxy).team_roll(team, season, team_df).iloc[2:].reset_index(drop=True)
                home_df = df[df['T'] == 'H'].reset_index(drop=True)
                results.append(home_df['R'])

                for i in range(len(home_df)):
                    opponent = home_df['Opponent'][i]
                    if opponent == 'Kangaroos':
                        opponent = 'North Melbourne'
                    print(team, opponent)
                    opp_df = History(mapping, proxy).team_roll(opponent, season, team_df)
                    rnd = home_df['Rnd'][i]
                    home = home_df[home_df['Rnd'] ==
                                   rnd][
                        ['Rnd', 'F_mean', 'F_std', 'A_mean', 'A_std', 'M_mean', 'A_std', 'W_sum', 'perc']].values
                    away = opp_df[opp_df['Rnd'] ==
                                  rnd][['F_mean', 'F_std', 'A_mean', 'A_std', 'M_mean', 'A_std', 'W_sum', 'perc']].values
                    team_hg.append(np.concatenate([home, away], axis=1)[0])
            y = [y for x in results for y in x]
            np.save(data_path + '/results-' + year + '.npy', y)
            np.save(data_path + '/training-' + year + '.npy', team_hg)

    def team_roll(self, team, season, team_df={}, shift=1, web=False):
        proxy = self.proxy
        mapping = self.mapping

        roll = 25
        if web is True:
            df = Scrape(mapping=mapping, proxy=proxy).scrape_history(team, season)
        else:
            df = team_df[team, season]
        hist = pd.DataFrame()
        hist['Team'] = np.full(len(df), team, dtype=object)
        hist['Rnd'] = np.array([s.replace('R', '') for s in df['Rnd']]).astype(int)
        hist['T'] = df['T']
        hist['Opponent'] = df['Opponent']
        result = np.where(df['R'] == 'W', 1, df['R'])
        result = np.where(result == 'D', 1, result)
        hist['R'] = np.where(result == 'L', 0, result)

        hist['F_mean'] = df['F'].shift(shift).rolling(roll, min_periods=1).mean()
        hist['A_mean'] = df['A'].shift(shift).rolling(roll, min_periods=1).mean()
        hist['M_mean'] = df['M'].shift(shift).rolling(roll, min_periods=1).mean()
        hist['W_sum'] = hist['R'].shift(shift).rolling(roll, min_periods=2).sum()
        hist['F_std'] = df['F'].shift(shift).rolling(roll, min_periods=1).std()
        hist['A_std'] = df['A'].shift(shift).rolling(roll, min_periods=1).std()
        hist['M_std'] = df['M'].shift(shift).rolling(roll, min_periods=1).std()
        hist['perc'] = hist['F_mean'] / hist['A_mean']
        return hist
