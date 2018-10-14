from sklearn.linear_model import SGDRegressor
from ParseData import ParseData


# metaprameters
numtrainingpoints = 10


def getrows(range):
    pass
    #return x_data, y_data


def iter_minibatches(chunksize):
    # Provide chunks one by one
    chunkstartmarker = 0
    while chunkstartmarker < numtrainingpoints:
        chunkrows = range(chunkstartmarker, chunkstartmarker + chunksize)
        X_chunk, y_chunk = getrows(chunkrows)
        yield X_chunk, y_chunk
        chunkstartmarker += chunksize


def main():
    ps = ParseData()
    x_t, y_t, x_v, y_v = ps.get_train_set()

    batcherator = iter_minibatches(chunksize=1000)
    model = SGDRegressor()

    print(model.fit(x_t, y_t))

    # Train model
    # for X_chunk, y_chunk in batcherator:
    #     model.partial_fit(X_chunk, y_chunk)

    # Now make predictions with trained model
    # y_predicted = model.predict(X_test)


if __name__ == "__main__":
    main()
