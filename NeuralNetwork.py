# Import
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from pylab import savefig
from random import randint
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

# Import data
data = pd.read_csv('NewNFA-Filtered.csv')

# Shuffle the dataset, before anything else.
data = data.sample(frac=1)

# Drop non-numerical versions of hot-encoded features.
data = data.drop('country', 1)
data = data.drop('UN_subregion', 1)
Y = data['carbon'].values
X = data.drop(['carbon'], 1).values

# Separate datasets into training and testing
x_train, y_train = X[:-730], Y[:-730]
x_test, y_test = X[-730:], Y[-730:]

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

X_data, Y_data = x_train, y_train

# accuracy and MSE accumulators
r2_train, mse_avg_train = 0.0, 0.0
r2_valid, mse_avg_valid = 0.0, 0.0
r2_test, mse_avg_test = 0.0, 0.0

# Plot initialization
plt.ion()

# K-fold cross validation
k = 5 # number of folds
kf = KFold(n_splits=k)
ki = 0
for ind_train, ind_valid in kf.split(x_train):
    ki += 1

    # get train and validation sets
    x_train, y_train = X_data[ind_train], Y_data[ind_train]
    x_valid, y_valid = X_data[ind_valid], Y_data[ind_valid]

    # Setup plot of the current fold
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    line1, = ax1.plot(y_valid, color='b', label='Y Validation')
    line2, = ax1.plot(y_valid * 0.5, color='r', label='Prediction', alpha=0.3)
    plt.legend()
    plt.show()

    # Number of observations in training data
    n0 = x_train.shape[1]

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

    beta = 0.1
    loss_reg = tf.reduce_mean(loss + beta * regularizer)

    # Optimizer
    opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_reg)

    # Init
    net.run(tf.global_variables_initializer())
    
    # Run
    epochs = 50
    batch_size = 100
    mse_train, mse_valid = [], []
    for e in range(epochs):
        print(e)

        # Minibatch training
        for i in range(0, len(y_train) // batch_size):
            start = i * batch_size
            batch_x = x_train[start:start + batch_size]
            batch_y = y_train[start:start + batch_size]

            # Run optimizer with batch
            net.run(opt, feed_dict={X: batch_x, Y: batch_y})

            # Show progress
            if np.mod(i, 50) == 0:
                # Observation & Prediction plot
                pred_valid = net.run(out, feed_dict={X: x_valid})
                line2.set_ydata(pred_valid)
                plt.title('Epoch ' + str(e) + ', Batch ' + str(i) + ', Fold ' + str(ki))
                plt.pause(0.01)

                # Used in Error vs Iteration plot
                mse_train.append(net.run(loss_reg, feed_dict={X: x_train, Y: y_train}))
                mse_valid.append(net.run(loss_reg, feed_dict={X: x_valid, Y: y_valid}))

    # Evaluate model once per fold
    # Train Set
    mse_avg_train += mse_train[-1]
    pred_train = net.run(out, feed_dict={X: x_train})
    r2_train += r2_score(y_train, pred_train[0])

    # Validation Set
    mse_avg_valid += mse_valid[-1]
    pred_valid = net.run(out, feed_dict={X: x_valid})
    r2_valid += r2_score(y_valid, pred_valid[0])
    
    # Test Set
    mse_avg_test = net.run(loss_reg, feed_dict={X: x_test, Y: y_test})
    pred_test = net.run(out, feed_dict={X: x_test})
    r2_test += r2_score(y_test, pred_test[0])

    # close interactive session
    net.close()

    plt.cla()
    plt.clf()

    # TODO: não consigo fazer plotar, dá erro.
    """plt.figure()
    plt.plot(mse_train)
    plt.title('Error vs number of iterations')
    plt.xlabel('Number of iterations')
    plt.ylabel('Error')
    plt.show()
    savefig('ErrorVsIteration' + str(ki) + '.png', bbox_inches='tight')
    plt.cla()
    plt.clf()"""
    plt.close()

# Print results
print('MSE Train: ', mse_avg_train / k)
print('MSE Validation: ', mse_avg_valid / k)
print('MSE Test: ', mse_avg_test / k)
print()
print('R2 Score Train: ', r2_train / k)
print('R2 Score Validation: ', r2_valid / k)
print('R2 Score Test: ', r2_test / k)