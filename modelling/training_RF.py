from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
        'max_depth': randint(low=3, high=20),
        'min_samples_split':  uniform(0.01, 0.99),
        'min_samples_leaf': randint(low=1, high=10),
        'min_weight_fraction_leaf': uniform(0, 0.5),
        'max_features': randint(low=1, high=18),
        'max_leaf_nodes': randint(low=2, high=1000),
        'min_impurity_decrease': uniform(0, 2)
    }

    spaceB = {
        'n_estimators': Integer(200, 1000),
        'max_depth': Integer(3, 20),
        'min_samples_split':  Real(0.01, .99, "uniform"),
        'min_samples_leaf': Integer(1, 10),
        'min_weight_fraction_leaf': Real(0, 0.5, "uniform"),
        'max_features': Integer(1, 17),
        'max_leaf_nodes': Integer(2, 1000),
        'min_impurity_decrease': Real(0, 2)
    }

    def trainR(self, X_list, y_list, space=spaceR, cv=5):
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
            classifier = RandomForestClassifier(n_jobs=-1)
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
                                     n_iter=n_calls, scoring=self.scorer, cv=cv, n_jobs=-1, iid=False)

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

    def trainB(self, X_list, y_list, n_points=1, space=spaceB, cv=5):
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
            classifier = RandomForestClassifier(n_jobs=-1)
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
                                scoring=self.scorer, cv=cv,
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
