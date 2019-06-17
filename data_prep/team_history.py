import pandas as pd
import numpy as np
from data_prep.web_scraping import Scrape


class History:
    def __init__(self, mapping, proxy, enc):
        """
        :param mapping:
        :param proxy:
        :param enc: fitted ordinal encoder
        """
        self.mapping = mapping
        self.proxy = proxy
        self.enc = enc

    def generate_team_history(self, season_list=range(1, 16)):
        """
        Collects the data form the web into dataframes. team_df is a dictionary of dataframes.
        :param season_list:
        :return:
        """
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

    def team_roll(self, team, season, team_df={}, shift=1, web=False):
        """
        Returns rolling stats for overall season
        :param team:
        :param season:
        :param team_df:
        :param shift:
        :param web:
        :return:
        """
        proxy = self.proxy
        mapping = self.mapping
        enc = self.enc
        grounds = enc.categories_[0]

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

        df = df.shift(shift)  # shift the entire df
        hist[['F_mean', 'A_mean', 'M_mean']] = df[['F', 'A', 'M']].rolling(roll, min_periods=1).mean()
        hist['R_mean'] = hist[['R']].shift(shift).rolling(roll, min_periods=1).mean()  # needs shifting here too
        hist[['F_std', 'A_std', 'M_std']] = df[['F', 'A', 'M']].rolling(roll, min_periods=1).std(ddof=0)
        hist['perc'] = hist['F_mean'] / hist['A_mean']

        # hist[['F_mean5', 'A_mean5', 'M_mean5']] = df[['F', 'A', 'M']].rolling(5, min_periods=1).mean()
        # hist['perc5'] = hist['F_mean5'] / hist['A_mean5']

        grnds = df[['Venue']]
        mask = np.isin(grnds, grounds)
        grnds.values[~mask] = 'other'
        hist['grnd'] = enc.transform(grnds).flatten()
        return hist

    def generate_game_data(self, data_path, team_df, season_list=range(1, 16)):
        """
        Generates the training data for overall season stats
        :param data_path:
        :param team_df:
        :param season_list:
        :return:
        """
        mapping = self.mapping
        proxy = self.proxy
        enc = self.enc

        for season in season_list:
            year = str(2019 - season)
            teams = list(mapping.keys())
            teams.remove('Kangaroos')

            if season >= 8:
                teams.remove('Greater Western Sydney')
            if season >= 9:
                teams.remove('Gold Coast')

            team_hg = []
            results = []

            for team in teams:
                print(year, team)
                df = History(mapping, proxy, enc).team_roll(team, season, team_df)
                home_df = df[df['T'] == 'H'].reset_index(drop=True)
                results.append(home_df['R'])

                home_cols = ['Rnd', 'F_mean', 'F_std', 'A_mean', 'A_std', 'M_mean', 'A_std', 'R_mean', 'perc']
                away_cols = ['F_mean', 'F_std', 'A_mean', 'A_std', 'M_mean', 'A_std', 'R_mean', 'perc', 'grnd']

                for i in range(len(home_df)):
                    opponent = home_df['Opponent'][i]
                    if opponent == 'Kangaroos':
                        opponent = 'North Melbourne'
                    opp_df = History(mapping, proxy, enc).team_roll(opponent, season, team_df)
                    rnd = home_df['Rnd'][i]
                    home = home_df[home_df['Rnd'] == rnd][home_cols].values
                    away = opp_df[opp_df['Rnd'] == rnd][away_cols].values
                    team_hg.append(np.concatenate([home, away], axis=1)[0])
            y = [y for x in results for y in x]
            np.save(data_path + '/results-' + year + '.npy', y)
            np.save(data_path + '/training-' + year + '.npy', team_hg)
