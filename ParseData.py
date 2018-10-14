import math

import pandas as pd


class ParseData(object):
    def __init__(self):
        self.validation_ratio = self.test_ratio = 15
        self.droplist = ["country", "ISO alpha-3 code", "UN_region","UN_subregion", "record", "carbon"]
        self.x_train, self.y_train, self.x_validation, self.y_validation, self.x_test, self.y_test = self._get_set()

    # TODO está usando isso ainda??
    # def shuffle(self):
    #     file = pd.read_csv("NFA 2018.csv")
    #     file = file.sample(frac=1)
    #     file.to_csv("NFA2018Shuffled.csv")

    def _get_set(self):
        file = pd.read_csv("NFA2018Shuffled.csv")
        y = file["carbon"]
        x = file.drop(file.columns[file.columns.str.contains("Unnamed", case=False)], axis=1)
        x = x.drop(self.droplist, axis=1).dropna()

        # Com os dados quebrados removidos, temos um numero menor de experimentos
        totallen = len(x)
        validation = math.floor(totallen * (self.validation_ratio / 100))
        test = math.floor(totallen * (self.test_ratio / 100))

        x_train = x.values[:-(validation + test)]
        y_train = y[:-(validation + test)]
        x_validation = x.values[(totallen - test - validation):(totallen - test)]
        y_validation = y[(totallen - test - validation):(totallen - test)]
        x_test = x.values[(totallen - test):]
        y_test = y[(totallen - test):]

        return x_train, y_train, x_validation, y_validation, x_test, y_test

    def get_train_set(self):
        return self.x_train, self.y_train, self.x_validation, self.y_validation

    def get_test_set(self):
        return self.x_test, self.y_test
