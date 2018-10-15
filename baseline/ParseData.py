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
	# Import data
	data = pd.read_csv('NewNFA-Filtered.csv')
	data = data.drop('country', 1)
	data = data.drop('UN_subregion', 1)
	cols = data.columns.tolist()

	# Dimensions of dataset
	n, m = data.shape

	# Separate datasets into training, validation and testing
	Y = data['carbon'].values
	X = data.drop(['carbon'], 1).values

	x_train, y_train = X[:-1460], Y[:-1460]
	x_valid, y_valid = X[-1460:-730:], Y[-1460:-730:]
	x_test, y_test = X[-730:], Y[-730:]

	print(x_train.shape)
	print(x_valid.shape)
	print(x_test.shape)

	return x_train, y_train, x_valid, y_valid, x_test, y_test


def get_train_set():
	x_train, y_train, x_valid, y_valid, x_test, y_test = get_set()

	return x_train, y_train, x_valid, y_valid


def get_test_set():
	x_train, y_train, x_valid, y_valid, x_test, y_test = get_set()

	return x_test, y_test