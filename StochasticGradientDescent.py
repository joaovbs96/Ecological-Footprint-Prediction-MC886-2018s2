import ParseData
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

def main():
    x_train, y_train, x_valid, y_valid = ParseData.get_train_set()
    x_test, y_test = ParseData.get_test_set()

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_valid = scaler.transform(x_valid)
    x_test = scaler.transform(x_test)
    x_train = np.insert(x_train, 0, 1, axis=1)
    x_valid = np.insert(x_valid, 0, 1, axis=1)
    x_test = np.insert(x_test, 0, 1, axis=1)
    regr = linear_model.SGDRegressor(max_iter=10000, tol=0.00000001, eta0=0.1)
    a = regr.fit(x_train, y_train)
    # Aplica os dados de validacao na base treinada com os dados de treino
    predicted = regr.predict(x_valid)
    print("Score Validation: " + str(r2_score(y_valid, predicted)))
    # Aplica os dados de teste na base treinada com os dados de treino
    predicted = regr.predict(x_test)
    print("Score test: " + str(r2_score(y_test, predicted)))


if __name__ == "__main__":
    main()

