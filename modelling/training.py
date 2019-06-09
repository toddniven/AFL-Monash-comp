from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from xgboost.sklearn import XGBClassifier
from skopt import BayesSearchCV
from data_prep.feature_eng import Features
from time import time

from skopt.space import Real, Integer
from scipy.stats import uniform
from scipy.stats import randint


class Training:
    def __init__(self, n_calls):
        self.n_calls = n_calls

    def afl_loss(y_true, y_pred):
        return -np.sum(1 + np.log2(y_true * y_pred + (1 - y_true) * (1 - y_pred)))
    scorer = make_scorer(afl_loss, greater_is_better=False, needs_proba=True)

    spaceR = {
        'n_estimators': randint(low=350, high=700),
        'max_depth': randint(low=3, high=6),
        'learning_rate': uniform(0.005, 0.1),
        'gamma': uniform(0.04, 0.05),
        'min_child_weight': randint(low=1, high=5),
        'scale_pos_weight': uniform(0, 2),
        'max_delta_step': randint(low=0, high=5),
        'colsample_bytree': uniform(0.1, 0.9),
        'colsample_bylevel': uniform(0.1, 0.9),
        'colsample_bynode': uniform(0.1, 0.9),
        'subsample': uniform(0.1, 0.9),
        'reg_lambda': uniform(0.0, 2.0),
        'reg_alpha': uniform(0.0, 2.0),
        'base_score': uniform(0.5, 0.1),
    }

    spaceB = {
        'n_estimators': Integer(200, 1000),
        'max_depth': Integer(3, 6),
        'learning_rate': Real(10 ** -4, 0.1, "log-uniform"),
        'gamma': Real(10 ** -5, 0.1, "log-uniform"),
        'min_child_weight': Integer(1, 5),
        'scale_pos_weight': Real(0, 2, "uniform"),
        'max_delta_step': Integer(0, 5),
        'colsample_bytree': Real(0.1, 1.0, "uniform"),
        'colsample_bylevel': Real(0.1, 1.0, "uniform"),
        'colsample_bynode': Real(0.1, 1.0, "uniform"),
        'subsample': Real(0.1, 1.0, "uniform"),
        'reg_lambda': Real(0.0, 2.0, "uniform"),
        'reg_alpha': Real(0.0, 2.0, "uniform"),
    }

    def trainR(self, X_list, y_list, space=spaceR):
        """
        RandomSearchCV method
        :param X_list: List of training sets
        :param y_list: List of targets
        :param space: parameter space
        :return: models an metrics
        """
        n_calls = self.n_calls

        scores = []
        val_scores = []
        best_models = []

        for j in range(len(X_list)):
            classifier = XGBClassifier(n_jobs=-1)
            y = y_list.copy()
            X = X_list.copy()
            y_test = y.pop(j)
            X_test = X.pop(j)
            y_train = np.concatenate(y, axis=0)
            X_train = np.concatenate(X, axis=0)

            X_train = Features().div_cols(X_train).values
            X_test = Features().div_cols(X_test).values

            start = time()

            opt = RandomizedSearchCV(classifier, param_distributions=space,
                                     n_iter=n_calls, scoring=self.scorer, cv=5, n_jobs=-1, iid=False)

            opt.fit(X_train, y_train)
            model = opt.best_estimator_
            print('Season', 2019 - j)
            print("Random CV search took %.2f seconds for %d candidates"
                  " parameter settings." % ((time() - start), n_calls))
            print("val. score:", opt.best_score_)
            print("test score:", opt.score(X_test, y_test))
            # print(model)
            print("")
            best_models.append(model)
            val_scores.append(opt.best_score_)
            scores.append(opt.score(X_test, y_test))
        return scores, val_scores, best_models

    def trainB(self, X_list, y_list, n_points=1, space=spaceB):
        """
        BayesianSearchCV method
        :param X_list: List of training sets
        :param y_list: List of targets
        :param space: parameter space
        :return: models an metrics
        """
        n_calls = self.n_calls

        scores = []
        val_scores = []
        best_models = []

        for j in range(len(X_list)):
            classifier = XGBClassifier(n_jobs=-1)
            y = y_list.copy()
            X = X_list.copy()
            y_test = y.pop(j)
            X_test = X.pop(j)
            y_train = np.concatenate(y, axis=0)
            X_train = np.concatenate(X, axis=0)

            X_train = Features().div_cols(X_train).values
            X_test = Features().div_cols(X_test).values

            start = time()
            opt = BayesSearchCV(classifier, search_spaces=space,
                                scoring=self.scorer, cv=5,
                                n_points=n_points,
                                n_iter=n_calls, n_jobs=-1)

            opt.fit(X_train, y_train)
            model = opt.best_estimator_
            print('Season', 2019 - j)
            print("Bayes CV search took %.2f seconds for %d candidates"
                  " parameter settings." % ((time() - start), n_calls))
            print("val. score:", opt.best_score_)
            print("test score:", opt.score(X_test, y_test))
            # print(model)
            print("")
            best_models.append(model)
            val_scores.append(opt.best_score_)
            scores.append(opt.score(X_test, y_test))
        return scores, val_scores, best_models
