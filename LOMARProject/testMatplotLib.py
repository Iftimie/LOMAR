import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from sklearn import cross_validation
import functions
import matplotlib.patches as mpatches
import os

X = np.load('X.npy')
Y = np.load('Y.npy')

from functions import splitTrainValidTest
X_train,Y_train,X_validation,Y_validation,X_test,Y_test = splitTrainValidTest(X,Y)

X_train = np.load('Xinc.npy')
Y_train = np.load('Yinc.npy')

scalerX = preprocessing.StandardScaler()
X_train = scalerX.fit_transform(X_train)
scalerY = preprocessing.StandardScaler()
Y_train = scalerY.fit_transform(Y_train)

X = scalerX.transform(X)
Y = scalerY.transform(Y)

X_test = scalerX.transform(X_test)
Y_test = scalerY.transform(Y_test)

X_validation = scalerX.transform(X_validation)
Y_validation = scalerY.transform(Y_validation)


# X_train, X_test, Y_train,Y_test = cross_validation.train_test_split(X, Y, test_size=0.183673, random_state=236)
# X_train, X_validation,Y_train,Y_validation = cross_validation.train_test_split(X_train, Y_train, test_size=0.125, random_state=236)
#
# scalerX = preprocessing.StandardScaler()
# X_train = scalerX.fit_transform(X_train)
# scalerY = preprocessing.StandardScaler()
# Y_train = scalerY.fit_transform(Y_train)


print (X.shape)
inSize = X.shape[1]
outSize = Y.shape[1]
hiddenSizes = [1200,400,100,15]

from kalmanFilter.kalman import KalmanFilter

kalman = KalmanFilter()

with tf.Session() as sess:
    learning_rate_placeholder, x_placeholder, y_placeholder, optimizer, cost, pred,keep_prob = functions.createNetwork(inSize=inSize,outSize=outSize,hiddenSizes=hiddenSizes)

    sess.run(tf.initialize_all_variables())
    new_saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.realpath(__file__)))
    if ckpt and ckpt.model_checkpoint_path:
        new_saver.restore(sess, os.path.dirname(os.path.realpath(__file__)) + "/")

    classification = sess.run(pred, feed_dict={x_placeholder: X_test, keep_prob: 1.0})
    classification_real = scalerY.inverse_transform(classification)
    Y_test = scalerY.inverse_transform(Y_test)
    functions.plotData([Y_test, classification_real])

    print("accuracy : ", functions.accuracy(Y_test, classification_real))
    error_list1 = functions.squared_error_list(Y_test, classification_real)

    classification_real_filtered = []

    for i in range(len(classification_real)):
        detectedPoints = [(classification_real[i, 0],classification_real[i, 1])]
        kalman.predict()
        kalman.update(detectedPoints)
        point = kalman.getPoint()
        classification_real_filtered.append(point)


    print ("accuracy filtered: ",functions.accuracy(Y_test,classification_real_filtered))
    error_list2 = functions.squared_error_list(Y_test,classification_real_filtered)
    red_patch = mpatches.Patch(color='red', label='Error of filtered coordinates')
    yellow_patch = mpatches.Patch(color='yellow', label='Error of neural network output coordinates')
    plt.legend(handles=[yellow_patch,red_patch])
    plt.xlabel('Index of the input values')
    plt.ylabel('Squared error')
    fit = np.polyfit(range(len(error_list1)), error_list1, 1)
    plt.plot(range(len(error_list1)), np.array(range(len(error_list1))) * fit[0] + fit[1], '-',color='#F7D358')
    fit = np.polyfit(range(len(error_list2)), error_list2, 1)
    plt.plot(range(len(error_list2)), np.array(range(len(error_list2))) * fit[0] + fit[1], '--',color='red')
    plt.plot(range(len(error_list1)), error_list1, 'y^',range(len(error_list2)), error_list2, 'r.')
    plt.savefig("color.png")
    from PIL import Image
    Image.open('color.png').convert('L').save('bw.png')
    plt.show()


    kalman = KalmanFilter()

    plt.axis([0, 40, 0, 4])
    plt.ion()
    plt.pause(2)
    print (classification_real[0])
    for i in range(len(classification_real)):
        y = np.random.random()
        detectedPoints = [(classification_real[i, 0],classification_real[i, 1])]
        kalman.predict()
        kalman.update(detectedPoints)
        point = kalman.getPoint()

        #if i % 20 ==0 :
        plt.scatter(classification_real[i, 0], classification_real[i, 1],marker='^', color="yellow", edgecolors="black")
        plt.scatter(point[0], point[1],marker='.', color="red", edgecolors="black")
        plt.scatter(Y_test[i, 0], Y_test[i, 1],marker='*', color="blue")
        plt.savefig('output/track' + str(i) + '.png')
        Image.open('output/track' + str(i) + '.png').convert('L').save('output/track' + str(i) + '.png')
        plt.pause(0.0005)
        plt.clf()
        plt.axis([0, 40, 0, 4])
        plt.ion()

################################### FROM FN 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

#output_x,y,size,dx,dy_olg_alg.csv
#output_x,y,size,dx,dy_new_alg.csv
x = np.loadtxt('output_x,y,size,dx,dy_olg_alg.csv', delimiter=',')
#image = np.zeros((480,640))
#fig,ax = plt.subplots(1)
#ax.imshow(image)
#for i in range(len(x)):
#    ax.add_patch(
#        patches.Rectangle(
#            (x[i,0], x[i,1]),   # (x,y)
#            x[i,2],          # width
#            x[i,2],          # height
#            fill=False
#        )
#    )
#plt.show()

fig = plt.figure()
plt.axis('equal')
plt.grid()
#plt.gca().invert_xaxis()
ax = fig.add_subplot(111)
ax.set_xlim(0, 640)
ax.set_ylim(0, 480)
ax.set_ylim(ax.get_ylim()[::-1])


#plt.axis([0, 640, 0, 480])
plt.ion()
old_scale = x[0,2]
for i in range(len(x)):
    ax.add_patch(patches.Rectangle((x[i,0], x[i,1]), x[i,2], x[i,2], fc='r',fill=False))
    if x[i,2] != old_scale:
        old_scale= x[i,2]
        plt.clf()
        plt.axis('equal')
        plt.grid()
        #plt.gca().invert_xaxis()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 640)
        ax.set_ylim(0, 480)
        ax.set_ylim(ax.get_ylim()[::-1])
        #plt.axis([0, 640, 0, 480])
    plt.pause(0.5)
    plt.ion()
####################################FROM FN

    while True:
        plt.pause(0.0005)
