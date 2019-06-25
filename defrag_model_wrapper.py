# -*- coding: utf-8 -*-
"""
@author: Satoshi Hara
@author: kzkymn
"""

import os
import tempfile

import numpy as np
import pandas as pd

from defragTrees import DefragModel


class DefragModelWrapper():
    def __init__(self):
        self.default_defrag_options = {
            "maxitr": 100,
            "qitr": 0,
            "tol": 1e-6,
            "restart": 20,
            "verbose": 0,
            "k_max": 10,
            "fittype": "FAB"
        }
        self.forest_ = None
        self.model_ = None
        self.X_colnames_ = None
        self.X_converted = False
        self.y_colname_ = None
        self.y_uniques_ = None
        self.y_converted = False

    @staticmethod
    # def __df_to_array(df: pd.DataFrame) -> np.ndarray, List:
    def __df_to_array(df: pd.DataFrame) -> np.ndarray:
        """convert pd.DataFrame object to np.ndarray for training.

        Arguments:
            df {pd.DataFrame} -- pd.DataFrame for conversion

        Returns:
            np.ndarray -- converted data
        """
        if not isinstance(df, np.ndarray):
            if isinstance(df, pd.DataFrame):
                df_colnames = df.columns.tolist()
                res = df.values
            else:
                raise TypeError(
                    "Type of DataFrame \"{}\" is not supported.", type(df))

            return res, df_colnames
        else:
            return df, None

    @staticmethod
    # def __series_to_array(series: pd.Series) -> np.ndarray, List, str:
    def __series_to_array(series: pd.Series) -> np.ndarray:
        """convert pd.Series object to np.ndarray for training.

        Arguments:
            series {pd.Series} -- pd.Series for conversion

        Returns:
            np.ndarray -- converted data
        """
        if not isinstance(series, np.ndarray):
            if isinstance(series, pd.DataFrame):
                colname = series.columns[0]
                res, series_uniques = pd.factorize(series[colname])
            elif isinstance(series, pd.Series):
                colname = series.name
                res, series_uniques = pd.factorize(series)
            else:
                raise TypeError("Type of X \"{}\" is not supported.", type())

            return res, series_uniques, colname
        else:
            return series, None, None

    def __encode_eval_sets(self, eval_sets):
        new_sets = []
        for X_test, y_test in eval_sets:
            X_test, _ = self.__df_to_array(X_test)
            if self.y_uniques_ is not None:
                y_test = self.y_uniques_.get_indexer(y_test)
            new_sets.append([X_test, y_test])

        return new_sets

    def fit(self, X, y, forest_class=None, forest=None,
            model_type: str = None, model_options: dict = {},
            fitting_options: dict = {},
            defrag_options: dict = {}):

        # convert X and y into np.ndarray if their types are pd.DataFrame
        X, self.X_colnames_ = self.__df_to_array(X)
        self.X_converted = False if self.X_colnames_ is None else True
        y, self.y_uniques_, self.y_colname_ = self.__series_to_array(y)
        self.y_converted = False if self.y_colname_ is None else True

        # instanciate forest object
        if forest is None:
            if forest_class is None:
                raise TypeError("model_obj and model_class are None.")
            forest = forest_class(**model_options)
            if "eval_set" in fitting_options:
                fitting_options["eval_set"] = self.__encode_eval_sets(
                    fitting_options["eval_set"])
            forest.fit(X, y, **fitting_options)

        self.forest_ = forest

        # infer type of model_class
        model_class_name = self.forest_.__class__.__name__
        if model_type is None or type(model_type) != str or model_type == "":
            if "Classifier" in model_class_name:
                model_type = "classification"
            elif "Regressor" in model_class_name:
                model_type = "regression"
            else:
                raise TypeError("Unexpected model_type {} has detected.".
                                format(model_class_name))

        # parse tree ensembles into the array of (feature index, threshold)
        with tempfile.TemporaryDirectory() as dname:
            if "XGB" in model_class_name:
                output_path = os.path.join(dname, "xgbmodel.txt")
                self.forest_.get_booster().dump_model(output_path)
                splitter = DefragModel.parseXGBtrees(output_path)
            elif "LGBM" in model_class_name:
                output_path = os.path.join(dname, "lgbmodel.txt")
                self.forest_.booster_.save_model(output_path)
                splitter = DefragModel.parseLGBtrees(output_path)
            elif "sklearn" in self.forest_.__class__.__module__:
                splitter = DefragModel.parseSLtrees(self.forest_)

            # fit simplified model
            if defrag_options is None or type(defrag_options) != dict\
                    or defrag_options == {}:
                defrag_options = self.default_defrag_options.copy()
            defrag_options["modeltype"] = model_type

            k_max = 10
            if "k_max" in defrag_options:
                new_k_max = defrag_options.pop("k_max")
                if new_k_max is not None and type(new_k_max) is int and\
                        new_k_max > 0:
                    k_max = new_k_max

            fittype = "FAB"
            if "fittype" in defrag_options:
                new_f_type = defrag_options.pop("fittype")
                if new_f_type is not None and type(new_f_type) is str and\
                        new_f_type in ["EM", "FAB"]:
                    fittype = new_f_type

            mdl = DefragModel(**defrag_options)
            mdl.fit(X, y, splitter, k_max, fittype=fittype)
            self.model_ = mdl
            return mdl

    def predict(self, X, y):
        if self.model_ is None:
            return None
        X, _ = self.__df_to_array(X)
        if self.y_uniques_ is not None:
            y = self.y_uniques_.get_indexer(y)

        return self.model_.predict(X, y)

    def evaluate(self, X, y):
        if self.model_ is None:
            return None
        X, _ = self.__df_to_array(X)
        if self.y_uniques_ is not None:
            y = self.y_uniques_.get_indexer(y)

        return self.model_.evaluate(X, y)

    def __str__(self):
        return str(self.model_)
