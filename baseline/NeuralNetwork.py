# Import
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

from pylab import savefig
from random import randint
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from ParseData import get_train_set, get_test_set

# # Import data
x_train, y_train, x_valid, y_valid = get_train_set()
x_test, y_test = get_test_set()

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
loss = tf.reduce_mean(tf.squared_difference(out, Y))

# Loss function using L2 Regularization
regularizer = tf.nn.l2_loss(W1)
regularizer += tf.nn.l2_loss(W2)
regularizer += tf.nn.l2_loss(W3)
regularizer += tf.nn.l2_loss(W4)
regularizer += tf.nn.l2_loss(W_out)

regularizer = tf.nn.l2_loss(B1)
regularizer += tf.nn.l2_loss(B2)
regularizer += tf.nn.l2_loss(B3)
regularizer += tf.nn.l2_loss(B4)
regularizer += tf.nn.l2_loss(bias_out)

beta = 10
loss_reg = tf.reduce_mean(loss + beta * regularizer)

# Optimizer
opt = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss_reg)

# Init
net.run(tf.global_variables_initializer())

# Setup plot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_valid, color='b', label='Y Validation')
line2, = ax1.plot(y_valid * 0.5, color='r', label='Prediction', alpha=0.3)
#line3, = ax1.plot(y_valid * 0.25)
plt.legend()
plt.show()

# Fit neural net
batch_size = 256
mse_train = []
mse_valid = []
# mse_test = []

# Run
epochs = 500
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
            mse_train.append(net.run(loss_reg, feed_dict={X: x_train, Y: y_train}))
            mse_valid.append(net.run(loss_reg, feed_dict={X: x_valid, Y: y_valid}))
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

plt.cla()
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.plot(mse_valid)
plt.title('Error vs number of iterations')
plt.xlabel('Number of iterations')
plt.ylabel('Error')
savefig('ErrorVsIteration.png', bbox_inches='tight')
