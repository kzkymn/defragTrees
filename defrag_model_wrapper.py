# -*- coding: utf-8 -*-
"""
@author: Satoshi Hara
@author: kzkymn
"""

import os

from defragTrees import DefragModel
import tempfile


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
        self.mdl_ = None

    def fit(self, X, y, forest_class=None, forest=None,
            model_type: str = None, model_options: dict = {},
            fitting_options: dict = {},
            defrag_options: dict = {}):

        # instanciate forest object
        if forest is None:
            if forest_class is None:
                raise TypeError("model_obj and model_class are None.")
            forest = forest_class(**model_options)
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
        return self.model_.predict(X, y)

    def evaluate(self, X, y):
        if self.model_ is None:
            return None
        return self.model_.evaluate(X, y)
