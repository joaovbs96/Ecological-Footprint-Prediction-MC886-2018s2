# Import
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from random import randint
from sklearn.metrics import r2_score

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

# Scale data
# scaler = MinMaxScaler(feature_range=(0, 1))
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_valid = scaler.transform(x_valid)
x_test = scaler.transform(x_test)

# Number of observations in training data
n0 = x_train.shape[1] # TODO: change variable name

# Neurons for each layer
n1, n2, n3, n4 = 1024, 512, 256, 128

# Session
net = tf.InteractiveSession()

# Placeholders
X = tf.placeholder(dtype=tf.float32, shape=[None, n0])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

# Initializers
sigma = 1
weights = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
biases = tf.zeros_initializer()

# Hidden weights
W1 = tf.Variable(weights([n0, n1]))
B1 = tf.Variable(biases([n1]))
W2 = tf.Variable(weights([n1, n2]))
B2 = tf.Variable(biases([n2]))
W3 = tf.Variable(weights([n2, n3]))
B3 = tf.Variable(biases([n3]))
W4 = tf.Variable(weights([n3, n4]))
B4 = tf.Variable(biases([n4]))

# Output weights
W_out = tf.Variable(weights([n4, 1]))
bias_out = tf.Variable(biases([1]))

# Hidden layer
H1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1))
H2 = tf.nn.relu(tf.add(tf.matmul(H1, W2), B2))
H2 = tf.nn.relu(tf.add(tf.matmul(H2, W3), B3))
H2 = tf.nn.relu(tf.add(tf.matmul(H2, W4), B4))

# Output layer (transpose!)
out = tf.transpose(tf.add(tf.matmul(H2, W_out), bias_out))

# Cost function
mse = tf.reduce_mean(tf.squared_difference(out, Y))

# Optimizer
opt = tf.train.AdamOptimizer().minimize(mse)

# Init
net.run(tf.global_variables_initializer())

# Setup plot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_valid)
line2, = ax1.plot(y_valid * 0.5)
line3, = ax1.plot(y_valid * 0.25)
plt.show()

# Fit neural net
batch_size = 256
mse_train = []
mse_valid = []
# mse_test = []

# Run
epochs = 5000
for e in range(epochs):

    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = x_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]

        # Run optimizer with batch
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})

        # Show progress
        if np.mod(i, 50) == 0:
            # MSE train and test
            mse_train.append(net.run(mse, feed_dict={X: x_train, Y: y_train}))
            mse_valid.append(net.run(mse, feed_dict={X: x_valid, Y: y_valid}))
            #mse_test.append(net.run(mse, feed_dict={X: x_test, Y: y_test}))

            print('MSE Validation: ', mse_valid[-1])
            #print('MSE Test: ', mse_test[-1])

            # Prediction
            pred = net.run(out, feed_dict={X: x_valid})
            #pred_test = net.run(out, feed_dict={X: x_test})

            print('R2 Score: ', r2_score(y_valid, pred[0]))

            ind = randint(0, len(y_valid)-1)
            print('i', i, ': ', 100*pred[0][i]/y_valid[i], '%')
            print('i', ind, ': ', 100*pred[0][ind]/y_valid[ind], '%')

            line2.set_ydata(pred)
            #line3.set_ydata(pred_test)

            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            plt.pause(0.01)

plt.clear()
plt.plot(mse_valid)
plt.title('Error vs iteration')
plt.show()
