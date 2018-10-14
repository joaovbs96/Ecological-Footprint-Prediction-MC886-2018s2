import ParseData
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

# So testando mesmo
regr = linear_model.SGDRegressor(alpha=100, max_iter=1000000, tol=0.001, eta0=1)