# -*- coding: utf-8 -*-
"""
@author: Satoshi Hara
@author: kzkymn
"""

import os
import tempfile
from typing import List

import numpy as np
import pandas as pd

from .defrag_trees import DefragModel


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
        self.model_type_ = None
        self.X_colnames_ = None
        self.X_converted_ = False
        self.y_colname_ = None
        self.y_uniques_ = None
        self.y_converted_ = False

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
    def __series_to_array(series: pd.Series, model_type: str = None):
        """convert pd.Series object to np.ndarray for training.

        Arguments:
            series {pd.Series} -- pd.Series for conversion

        Keyword Arguments:
            model_type {str} -- classification or regression (default: {None})

        Raises:
            ValueError: raises when the value of model_type is neither
            classification nor regression.
            TypeError: raises when the type of series is neither pd.DataFrame
            nor pd.Series.

        Returns:
            [type] -- converted data
        """
        if model_type is None or not isinstance(model_type, str)\
                or model_type == "":
            raise ValueError("model_type is not assigned.")
        if model_type != "classification" and model_type != "regression":
            raise ValueError("unrecognized model type {}.".format(model_type))

        series_uniques = None
        if not isinstance(series, np.ndarray):
            if isinstance(series, pd.DataFrame):
                colname = series.columns[0]
                if model_type == "classification":
                    res, series_uniques = pd.factorize(series[colname])
                else:
                    res = series[colname].values
            elif isinstance(series, pd.Series):
                colname = series.name
                if model_type == "classification":
                    res, series_uniques = pd.factorize(series)
                else:
                    res = series.values
            else:
                raise TypeError(
                    "Type of X \"{}\" is not supported.", type(series))

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

        # instanciate forest object
        if forest is None:
            if forest_class is None:
                raise TypeError("model_obj and model_class are None.")
            forest = forest_class(**model_options)

        # infer type of model_class
        model_class_name = forest.__class__.__name__
        if model_type is None or type(model_type) != str or model_type == "":
            if "Classifier" in model_class_name:
                self.model_type_ = "classification"
            elif "Regressor" in model_class_name:
                self.model_type_ = "regression"
            else:
                raise TypeError("Unexpected model_type {} has detected.".
                                format(model_class_name))

        # convert X and y into np.ndarray if their types are pd.DataFrame
        X, self.X_colnames_ = self.__df_to_array(X)
        self.X_converted_ = False if self.X_colnames_ is None else True
        y, self.y_uniques_, self.y_colname_ = self.__series_to_array(
            y, self.model_type_)
        self.y_converted_ = False if self.y_colname_ is None else True

        # fitting forest
        if "eval_set" in fitting_options:
            fitting_options["eval_set"] = self.__encode_eval_sets(
                fitting_options["eval_set"])
        forest.fit(X, y, **fitting_options)

        self.forest_ = forest

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
            defrag_options["modeltype"] = self.model_type_

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
        if self.model_type_ == "classification":
            if self.y_uniques_ is not None:
                y = self.y_uniques_.get_indexer(y)
        else:
            y, _, _ = self.__series_to_array(y, self.model_type_)

        return self.model_.evaluate(X, y)

    def __str__(self):
        if not self.X_converted_ and not self.y_converted_:
            return str(self.model_)
        else:
            return self.replace_str()

    def replace_str(self) -> str:
        """replace 'x_i' and 'y' into the original feature and target names.

        Returns:
            str -- a string representation of the wrapper
            where the feature and target names have been replaced
            into the original ones.
        """
        orig_str_lines = str(self.model_).split("\n")
        res = []
        for line in orig_str_lines:
            if line.strip().startswith("["):
                res.append(line)
            elif line.strip().startswith("y"):
                if self.y_converted_:
                    res.append(self.replace_yname(line,
                                                  self.y_colname_,
                                                  self.y_uniques_))
                else:
                    res.append(line)
            elif "x_" in line:
                if self.X_converted_:
                    res.append(self.replace_xname(line,
                                                  self.X_colnames_))
                else:
                    res.append(line)
            else:
                res.append(line)

        return "\n".join(res)

    def replace_xname(self, line: str, orig_names: List[str]) -> str:
        found_x = False
        found_bar = False
        found_idx_digit = False
        digit_lst = []
        res = []
        for c in line:
            if c == "x":
                found_x = True
            elif c == "_":
                if found_x:
                    found_bar = True
                else:
                    res.append(c)
            elif ord(c) >= ord("0") and ord(c) <= ord("9"):
                if found_x and found_bar:
                    found_idx_digit = True
                    digit_lst.append(c)
                else:
                    res.append(c)
            else:
                if found_x and found_bar and found_idx_digit:
                    idx = int("".join(digit_lst))
                    res.append(orig_names[idx-1])
                    found_x = found_bar = found_idx_digit = False
                    digit_lst = []
                res.append(c)

        if len(digit_lst) > 0:
            idx = int("".join(digit_lst))
            res.append(orig_names[idx-1])

        return "".join(res)

    def replace_yname(self, line: str, orig_name: str,
                      orig_values: List[str]) -> str:
        digit_lst = []
        res = []
        for c in line:
            if c == "y":
                res.append(orig_name)
            elif ord(c) >= ord("0") and ord(c) <= ord("9"):
                if self.model_type_ == "classification":
                    digit_lst.append(c)
                else:
                    res.append(c)
            else:
                if len(digit_lst) > 0:
                    idx = int("".join(digit_lst))
                    res.append(orig_values[idx])
                    digit_lst = []
                res.append(c)

        if len(digit_lst) > 0:
            idx = int("".join(digit_lst))
            res.append(orig_values[idx])

        res = [str(i) for i in res]
        return "".join(res)
