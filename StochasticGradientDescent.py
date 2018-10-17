import ParseData
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

def main():
    x_train, y_train, x_valid, y_valid = ParseData.get_train_set()
    x_test, y_test = ParseData.get_test_set()

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_valid = scaler.transform(x_valid)
    x_test = scaler.transform(x_test)
    regr = linear_model.SGDRegressor(alpha=100, max_iter=1000000, tol=0.0000001, eta0=0.001)
    a = regr.fit(x_train, y_train)
    # Aplica os dados de validacao na base treinada com os dados de treino
    predicted = regr.predict(x_valid)
    print("Error Validation: " + str(r2_score(y_valid, predicted)))
    # Aplica os dados de teste na base treinada com os dados de treino
    predicted = regr.predict(x_test)
    print("Error test: " + str(r2_score(y_test, predicted)))

if __name__ == "__main__":
    main()

