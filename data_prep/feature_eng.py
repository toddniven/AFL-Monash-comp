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

    def cols(self):
        return ['Rnd', 'h_F_mean', 'h_F_std', 'h_A_mean', 'h_A_std', 'h_M_mean', 'h_M_std', 'h_R_mean', 'h_perc',
                'a_F_mean', 'a_F_std', 'a_A_mean', 'a_A_std', 'a_M_mean', 'a_M_std', 'a_R_mean', 'a_perc']

