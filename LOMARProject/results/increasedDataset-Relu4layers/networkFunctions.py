import numpy as np
import tensorflow as tf

def createNetwork():
    # Parameters
    learning_rate = 0.000001
    dropout_rate = 0.5
    # Network Parameters
    n_hidden_1 = 32  # 1st layer number of features
    n_hidden_2 = 100  # 2nd layer number of features
    n_hidden_3 = 50
    n_hidden_4 = 20
    n_input = 309#X_train.shape[1]
    n_classes = 3
    # tf Graph input
    x = tf.placeholder("float", [None, 309])
    y = tf.placeholder("float", [None, 3])
    np.set_printoptions(precision=2)

    # Create model
    def multilayer_perceptron(x, weights, biases):
        # Hidden layers with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
        layer_3 = tf.nn.relu(layer_3)
        layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
        layer_4 = tf.nn.relu(layer_4)
        out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
        return out_layer

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], 0, 0.1)),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], 0, 0.1)),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], 0, 0.1)),
        'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], 0, 0.1)),
        'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes], 0, 0.1))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1], 0, 0.1)),
        'b2': tf.Variable(tf.random_normal([n_hidden_2], 0, 0.1)),
        'b3': tf.Variable(tf.random_normal([n_hidden_3], 0, 0.1)),
        'b4': tf.Variable(tf.random_normal([n_hidden_4], 0, 0.1)),
        'out': tf.Variable(tf.random_normal([n_classes], 0, 0.1))
    }
    new_saver = tf.train.Saver()
    pred = multilayer_perceptron(x, weights, biases)
    cost = tf.reduce_mean(tf.square(pred - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    return x,y,new_saver,optimizer,cost,pred