import numpy as np
import pandas as pd


class Features:
    def __init__(self):
        return None

    def div_cols(self, X):
        df = pd.DataFrame(X)
        for i in range(1, 9):
            df[32 + i] = df[i] / df[i + 8].astype(float)
            df[40 + i] = df[16 + i] / df[16 + i + 8].astype(float)
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)
        return df.values

    def training_cols(self):
        cols = ['Rnd', 'h_F_mean', 'h_F_std', 'h_A_mean', 'h_A_std', 'h_M_mean', 'h_M_std', 'h_W_sum', 'h_perc',
            'a_F_mean', 'a_F_std', 'a_A_mean', 'a_A_std', 'a_M_mean', 'a_M_std', 'a_W_sum', 'a_perc',
            'h_F_mean_hva', 'h_F_std_hva', 'h_A_mean_hva', 'h_A_std_hva', 'h_M_mean_hva', 'h_M_std_hva',
            'h_W_sum_hva', 'h_perc_hva',
            'a_F_mean_hva', 'a_F_std_hva', 'a_A_mean_hva', 'a_A_std_hva', 'a_M_mean_hva', 'a_M_std_hva',
            'a_W_sum_hva', 'a_perc_hva']
        return cols + ['F_mean', 'F_std', 'A_mean', 'A_std', 'M_mean', 'M_std', 'W_sum', 'perc',
                                'F_mean_hva', 'F_std_hva', 'A_mean_hva', 'A_std_hva', 'M_mean_hva',
                                'M_std_hva', 'W_sum_hva', 'perc_hva']

