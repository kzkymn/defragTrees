# -*- coding: utf-8 -*-
"""
@author: Satoshi Hara
@author: kzkymn
"""

import sys

import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


try:
    from defrag_model_wrapper import DefragModelWrapper
except ImportError:
    sys.path.append("../")
    from defrag_model_wrapper import DefragModelWrapper

# load classification data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target_names[iris.target], name="species")

orig_X, orig_y = load_iris(return_X_y=True)
assert (X.values == orig_X).all()
assert (pd.factorize(y)[0] == orig_y).all()
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train = X_test = X
y_train = y_test = y
# X_train = X_test = orig_X
# y_train = y_test = orig_y

model_settings = []
sklearn_setting = {"forest_class": RandomForestClassifier,
                   "model_options": {"n_estimators": 100,
                                     "min_samples_leaf": 10,
                                     "random_state": 0},
                   "fitting_options": {}}
model_settings.append(sklearn_setting)

lgbm_setting = {"forest_class": LGBMClassifier,
                "model_options": {"objective": "binary",
                                  "num_leaves": 31,
                                  "learning_rate": 0.05,
                                  "n_estimators": 50,
                                  "random_state": 0},
                "fitting_options": {"eval_set": [(X_test, y_test)],
                                    "eval_metric": "multi_logloss",
                                    "early_stopping_rounds": 10}}
model_settings.append(lgbm_setting)

# Setting early_stopping_rounds properly may be important
# to avoid errors when parsing XGB trees in defragTrees.
# I suspect the errors occur when there is a booster
# which has only one leaf like below.
# booster[174]:
# 0:leaf=-0.00331423
xgb_setting = {"forest_class": XGBClassifier,
               "model_options": {"max_depth": 4, "eta": 0.3, "random_state": 0,
                                 "silent": 1},
               "fitting_options": {"early_stopping_rounds": 10,
                                   "eval_set": [(X_test, y_test)]}}
model_settings.append(xgb_setting)

# train tree ensemble
for idx in range(len(model_settings)):
    model_setting = model_settings[idx]
    print("[[ learning by {} ]]".format(model_setting["forest_class"]))
    mdl_wrapper = DefragModelWrapper()
    mdl = mdl_wrapper.fit(X_train, y_train, **model_setting)
    score, cover, coll = mdl_wrapper.evaluate(X_test, y_test)

    # results
    print()
    print("<< defragTrees >>")
    print("----- Evaluated Results -----")
    print("Test Error = %f" % (score,))
    print("Test Coverage = %f" % (cover,))
    print("Overlap = %f" % (coll,))
    print()
    print("----- Found Rules -----")
    print(mdl_wrapper)


# load regression data
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target, name="medv")
# y = pd.DataFrame(boston.target, columns=("medv", ))

orig_X, orig_y = load_boston(return_X_y=True)
assert (X.values == orig_X).all()
assert (y.values == orig_y).all()
# assert (y["medv"].values == orig_y).all()
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train = X_test = X
y_train = y_test = y
# X_train = X_test = orig_X
# y_train = y_test = orig_y

model_settings = []
sklearn_setting = {"forest_class": RandomForestRegressor,
                   "model_options": {"n_estimators": 100,
                                     "min_samples_leaf": 10,
                                     "random_state": 0},
                   "fitting_options": {}}
model_settings.append(sklearn_setting)

lgbm_setting = {"forest_class": LGBMRegressor,
                "model_options": {"objective": "regression",
                                  "num_leaves": 31,
                                  "learning_rate": 0.05,
                                  "n_estimators": 50,
                                  "random_state": 0},
                "fitting_options": {"eval_set": [(X_test, y_test)],
                                    "eval_metric": "rmse",
                                    "early_stopping_rounds": 10}}
model_settings.append(lgbm_setting)

xgb_setting = {"forest_class": XGBRegressor,
               "model_options": {"max_depth": 4, "eta": 0.3, "random_state": 0,
                                 "silent": 1},
               "fitting_options": {"early_stopping_rounds": 10,
                                   "eval_set": [(X_test, y_test)]}}
model_settings.append(xgb_setting)

# train tree ensemble
for idx in range(len(model_settings)):
    model_setting = model_settings[idx]
    print("[[ learning by {} ]]".format(model_setting["forest_class"]))
    mdl_wrapper = DefragModelWrapper()
    mdl = mdl_wrapper.fit(X_train, y_train, **model_setting)
    score, cover, coll = mdl_wrapper.evaluate(X_test, y_test)

    # results
    print()
    print("<< defragTrees >>")
    print("----- Evaluated Results -----")
    print("Test Error = %f" % (score,))
    print("Test Coverage = %f" % (cover,))
    print("Overlap = %f" % (coll,))
    print()
    print("----- Found Rules -----")
    print(mdl_wrapper)
