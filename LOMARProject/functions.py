import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import tensorflow as tf

def plotSet(dataset):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = dataset[:, 0]
    ys = dataset[:, 1]
    zs = np.array([0 for value in range(len(dataset))])
    ax.scatter(xs, ys, zs, c='b', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def splitTrainValidTest(X,Y):
    Xtrain = []
    Ytrain = []
    Xvalid = []
    Yvalid = []
    Xtest = []
    Ytest = []
    print (len(Y))
    for i in range(0, len(Y)-10, 10):
        for j in range(i, i + 8):
            Xtrain.append(X[j])
            Ytrain.append(Y[j])
        Xvalid.append(X[i+8])
        Yvalid.append(Y[i+8])
        Xtest.append(X[i+9])
        Ytest.append(Y[i+9])
    return Xtrain,Ytrain,Xvalid,Yvalid,Xtest,Ytest

#XGood,YGood = increaseData(XGood,YGood,4,0.2,2)
def increaseData(XGood,YGood,multiplier,sigmaY,sigmaX):
    size = len(YGood)
    for i in range(size):
        for j in range(multiplier):
            new_target = YGood[i] + sigmaY * np.random.randn(1,2)# thats the matrix size
            #new_target[0][2] = YGood[i][2]
            YGood = np.concatenate((YGood, new_target), axis=0)
            new_measurement = XGood[i] + sigmaX * np.random.randn(1,172)
            XGood =np.concatenate((XGood,new_measurement),axis=0)
    return XGood,YGood

def createNetwork(inSize,outSize,hiddenSizes):
    # Parameters
    # dropout_rate = 0.5 #rarely used

    # tf Graph input and output
    learning_rate = tf.placeholder(tf.float32, shape=[])
    x = tf.placeholder("float", [None, inSize])
    y = tf.placeholder("float", [None, outSize])
    np.set_printoptions(precision=2)
    keep_prob = tf.placeholder(tf.float32)

    # Create model
    def multilayer_perceptron(x):
        # Hidden layers with RELU activation
        h1 = tf.get_variable(name="W1", shape=[inSize, hiddenSizes[0]],initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable(name="b1", shape=[hiddenSizes[0]], initializer=tf.contrib.layers.xavier_initializer())
        layer = tf.nn.elu(tf.add(tf.matmul(x, h1), b1))

        for i in range(len(hiddenSizes) - 1):  # for the rest of the hidden layers
            h = tf.get_variable(name="W1"+str(i),shape=[hiddenSizes[i], hiddenSizes[i + 1]], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name="b1"+str(i),shape=[hiddenSizes[i + 1]], initializer=tf.contrib.layers.xavier_initializer())
            layer = tf.nn.elu(tf.add(tf.matmul(layer, h), b))

        hout = tf.get_variable(name="Wl",shape=[hiddenSizes[-1], outSize], initializer=tf.contrib.layers.xavier_initializer())
        bout = tf.get_variable(name="bl",shape=[outSize], initializer=tf.contrib.layers.xavier_initializer())
        out_layer = tf.add(tf.matmul(layer, hout), bout)
        out_layer = tf.nn.dropout(out_layer, keep_prob)

        return out_layer

    pred = multilayer_perceptron(x)

    cost = tf.reduce_mean(tf.square(pred - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    return learning_rate, x, y, optimizer, cost, pred,keep_prob

def plotData(predictedAndGrounfTruth):
    fig = plt.figure('%d' % (0))
    ax = fig.add_subplot(111, projection='3d')
    xs1 = [value[0] for value in predictedAndGrounfTruth[0]]# 0 is ground 1 is predicted
    ys1 = [value[1] for value in predictedAndGrounfTruth[0]]
    #zs1 = [value[2] for value in listOfTuplesOfArrays[x][0]]
    zs1 = [0 for value in range(len(predictedAndGrounfTruth[0]))]#because octavians set does not have multiple floors
    ax.ticklabel_format(useOffset=False)
    ax.scatter(xs1, ys1, zs1, c='b', marker='o')  # c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]

    xs2 = [value[0] for value in predictedAndGrounfTruth[1]]
    ys2 = [value[1] for value in predictedAndGrounfTruth[1]]
    #zs2 = [value[2] for value in listOfTuplesOfArrays[x][1]]
    zs2 = [0 for value in range(len(predictedAndGrounfTruth[0]))] #because octavians set does not have multiple floors
    ax.scatter(xs2, ys2, zs2, c='r', marker='^')

    XLine = []
    for x1, x2 in zip(xs1, xs2):
        XLine.append([x1, x2])
    YLine = []
    for y1, y2 in zip(ys1, ys2):
        YLine.append([y1, y2])
    ZLine = []
    for z1, z2 in zip(zs1, zs2):
        ZLine.append([z1, z2])
    for XX, YY, ZZ in zip(XLine, YLine, ZLine):
        ax.plot(XX, YY, ZZ, label='connections', color='g')
        # ax.plot(XX, YY, ZZ, label='connections', color='g')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def accuracy(Y_real,Y_pred):
    if len(Y_real) != len(Y_pred):
        raise Exception('length not equal')
    bad = 0
    for i in range(len(Y_real)):
        if np.sqrt(np.sum(np.square(Y_pred[i]-Y_real[i])))>1:
            bad = bad+1
    return 1.- float(bad)/float(len(Y_real))

def squared_error_list(Y_real,Y_pred):
    if len(Y_real) != len(Y_pred):
        raise Exception('length not equal')
    error_list = []
    for i in range(len(Y_real)):
        error_list.append(np.sqrt(np.sum(np.square(Y_pred[i]-Y_real[i]))))
    return error_list




