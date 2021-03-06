import numpy as np
import pandas as pd


class Features:
    def __init__(self):
        return None

    def cols(self):
        return ['Rnd', 'h_F_mean', 'h_F_std', 'h_A_mean', 'h_A_std', 'h_M_mean', 'h_M_std', 'h_R_mean', 'h_perc',
                'a_F_mean', 'a_F_std', 'a_A_mean', 'a_A_std', 'a_M_mean', 'a_M_std', 'a_R_mean', 'a_perc',
                'grnd']

    def div_cols(self, X):
        df = pd.DataFrame(X)
        df.columns = Features().cols()
        df['perc'] = df['h_perc'] / df['a_perc']
        df['R_mean'] = df['h_R_mean'] / df['a_R_mean']
        df['F_ph_na'] = (df['h_F_mean'] + df['h_F_std']) / (df['a_F_mean'] - df['a_F_std'])
        df['F_ph_pa'] = (df['h_F_mean'] + df['h_F_std']) / (df['a_F_mean'] + df['a_F_std'])
        df['F_nh_na'] = (df['h_F_mean'] - df['h_F_std']) / (df['a_F_mean'] - df['a_F_std'])
        df['F_nh_pa'] = (df['h_F_mean'] - df['h_F_std']) / (df['a_F_mean'] + df['a_F_std'])
        df['A_nh_pa'] = (df['h_A_mean'] - df['h_A_std']) / (df['a_A_mean'] + df['a_A_std'])
        df['A_nh_na'] = (df['h_A_mean'] - df['h_A_std']) / (df['a_A_mean'] - df['a_A_std'])
        df['A_ph_pa'] = (df['h_A_mean'] + df['h_A_std']) / (df['a_A_mean'] + df['a_A_std'])
        df['A_ph_na'] = (df['h_A_mean'] + df['h_A_std']) / (df['a_A_mean'] - df['a_A_std'])
        df = df.drop(['h_F_std', 'h_A_std', 'h_M_std', 'a_F_std', 'a_A_std', 'a_M_std'], axis=1)
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)
        return df
