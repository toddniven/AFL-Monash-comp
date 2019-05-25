from sklearn.metrics import make_scorer
import numpy as np
from xgboost.sklearn import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from data_prep.feature_eng import Features
from time import time


class Training:
    def __init__(self, n_calls, cutoff_score):
        self.n_calls = n_calls
        self.cutoff_score = cutoff_score

    def train(self, X_list, y_list):
        n_calls = self.n_calls
        cutoff_score = self.cutoff_score

        def afl_loss(y_true, y_pred):
            return -np.sum(1 + np.log2(y_true * y_pred + (1 - y_true) * (1 - y_pred)))
        scorer = make_scorer(afl_loss, greater_is_better=False, needs_proba=True)

        scores = []
        best_models = []

        space = {
            'n_estimators': Integer(100, 500),
            'max_depth': Integer(3, 5),
            'learning_rate': Real(10 ** -3, 0.1, "log-uniform"),
            'colsample_bytree': Real(0.1, 1.0, "uniform"),
            'colsample_bylevel': Real(0.1, 1.0, "uniform"),
            'colsample_bynode': Real(0.1, 1.0, "uniform"),
            'subsample': Real(0.1, 1.0, "uniform"),
            'reg_lambda': Real(0.0, 1.0, "uniform"),
            'reg_alpha': Real(0.0, 1.0, "uniform"),
        }

        for j in range(len(X_list)):
            classifier = XGBClassifier(base_score=0.57574568288854, n_jobs=-1)
            y = y_list.copy()
            X = X_list.copy()
            y_test = y.pop(j)
            X_test = X.pop(j)
            y_train = np.concatenate(y, axis=0)
            X_train = np.concatenate(X, axis=0)

            X_train = Features().div_cols(X_train)
            X_test = Features().div_cols(X_test)

            start = time()
            opt = BayesSearchCV(classifier, search_spaces=space, scoring=scorer, cv=5, n_iter=n_calls, n_jobs=-1)

            # callback handler
            def on_step(iteration):
                score = opt.best_score_
                if score > cutoff_score:
                    print('Interrupting!')
                    return True

            opt.fit(X_train, y_train, callback=on_step)
            model = opt.best_estimator_
            print('Season', 2018 - j)
            print("Bayes CV search took %.2f seconds for %d candidates"
                  " parameter settings." % ((time() - start), n_calls))
            print("val. score:", opt.best_score_)
            print("test score:", opt.score(X_test, y_test))
            print(model)
            print("")
            best_models.append(model)
            scores.append(opt.score(X_test, y_test))
        return scores, best_models
