import pandas as pd


def shuffle():
	file = pd.read_csv("NFA 2018.csv")
	file = file.sample(frac=1)
	file.to_csv("NFA2018Shuffled.csv")


"""
Train -> 60914 rows
Validation -> 13053 rows
Test -> 13053 rows
"""


def get_set():
	file = pd.read_csv("NFA2018Shuffled.csv")

	y = file["carbon"]
	
	x = file.drop(file.columns[file.columns.str.contains("Unnamed", case = False)], axis=1)
	x = x.drop(["carbon"], axis=1)

	x_train, y_train = x.values[:-26106], y[:-26106]
	x_validation, y_validation = x.values[60914:73967], y[60914:73967]
	x_test, y_test = x.values[73967:], y[73967:]

	return x_train, y_train, x_validation, y_validation, x_test, y_test


def get_train_set():
	x_train, y_train, x_validation, y_validation, x_test, y_test = get_set()

	return x_train, y_train, x_validation, y_validation


def get_test_set():
	x_train, y_train, x_validation, y_validation, x_test, y_test = get_set()

	return x_test, y_test
    

# x_train, y_train, x_validation, y_validation = get_train_set()
# x_test, y_test = get_test_set()

# print(x_train.shape)
# print(y_train.shape)
# print(x_validation.shape)
# print(y_validation.shape)
# print(x_test.shape)
# print(y_test.shape)