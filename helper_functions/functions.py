import numpy as np
from data_prep.feature_eng import Features


class Helpers:
    def __init__(self):
        return None

    def averagingImp(self, models=[]):
        predictions = np.column_stack([
            model.feature_importances_ for model in models
        ])
        return np.mean(predictions, axis=1)

    def averagingModels(self, X, models=[]):
        predictions = np.column_stack([
            model.predict_proba(Features().div_cols(X).values)[:, 1] for model in models
        ])
        return np.mean(predictions, axis=1)

    def afl_loss(self, y_true, y_pred):
        return np.sum(1 + np.log2(y_true * y_pred + (1 - y_true) * (1 - y_pred)))
