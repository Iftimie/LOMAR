import numpy as np
import tensorflow as tf
from sklearn import cross_validation
from sklearn import decomposition
from sklearn import preprocessing
from usefulFunctions.functions import loadMeasurements,averageTargets,increaseData,saveData,plotData,plotSet
from usefulFunctions.functions import separateFloorsAndPrintAccuracy
from usefulFunctions.networkFunctions import createNetworkForPCA

# good_list_of_measurementsX,good_list_of_measurementsY = loadMeasurements(minStrength = 0)
# XGood,YGood = averageTargets(good_list_of_measurementsX,good_list_of_measurementsY)
#
# XGood = preprocessing.scale(XGood)
#
# X_train1, X_test, Y_train1, Y_test = cross_validation.train_test_split(XGood, YGood, test_size=0.183673, random_state=3678424)
#
# X_train, X_validation, Y_train,Y_validation = cross_validation.train_test_split(X_train1, Y_train1, test_size=0.125, random_state=753458)
#
# #X_train, Y_train = increaseData(X_train, Y_train,15,0.3,1.5)
#
# pca = decomposition.PCA(n_components=68)
# pca.fit(X_train)
# X_train1 = pca.transform(X_train1)
# X_train = pca.transform(X_train)
# X_validation = pca.transform(X_validation)
# X_test = pca.transform(X_test)
#
# saveData([X_train,X_test,X_validation,Y_train,Y_test,Y_validation],['X_train.npy','X_test.npy','X_validation.npy','Y_train.npy','Y_test.npy','Y_validation.npy'])
X_train =np.load('X_train.npy')
Y_train =np.load('Y_train.npy')
X_test =np.load('X_test.npy')
Y_test =np.load('Y_test.npy')
X_validation = np.load('X_validation.npy')
Y_validation =np.load('Y_validation.npy')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure('%d' % (0))
ax = fig.add_subplot(111, projection='3d')
xs1 = [value[0] for value in Y_train]  # 0 is ground 1 is predicted
ys1 = [value[1] for value in Y_train]
zs1 = [value[2] for value in Y_train]
ax.ticklabel_format(useOffset=False)
ax.scatter(xs1, ys1, zs1, c='b', marker='o')  # c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]

xs2 = [value[0] for value in Y_test]
ys2 = [value[1] for value in Y_test]
zs2 = [value[2] for value in Y_test]
ax.scatter(xs2, ys2, zs2, c='r', marker='^')
plt.show()

learningRate,x,y,new_saver,optimizer,cost,pred = createNetworkForPCA(X_train)
hit = 0
LR = 0.0001

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    ckpt = tf.train.get_checkpoint_state('/home/iftimie/PycharmProjects/SirajRavalTutorials/')
    if ckpt and ckpt.model_checkpoint_path:
        new_saver.restore(sess, ckpt.model_checkpoint_path)

    lastTrainCost = 9999
    batch_size = 10
    # Training cycle
    for epoch in range(0):
        avg_cost = 0.
        total_batch = int(X_train.shape[0]/batch_size) #X_train.shape[0] is total lenght (1400 or 490)
        # Loop over all batches
        for i in range(total_batch-1):
            batch_x = X_train[i*batch_size:(i+1)*batch_size]
            batch_y = Y_train[i*batch_size:(i+1)*batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, p = sess.run([optimizer, cost, pred], feed_dict={learningRate:LR ,x: batch_x, y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch

        if epoch % 40 == 0:
            train = sess.run(pred, feed_dict={x: X_train})
            trainCost = 0
            for predictedTrain, groundTruthTrain in zip(train, Y_train):
                trainCost += (predictedTrain - groundTruthTrain).dot(predictedTrain - groundTruthTrain) / train.shape[0]
            print("cost for train %.10f" % (trainCost))
            lastTrainCost = trainCost
            validation = sess.run(pred, feed_dict={x: X_validation})
            validationCost = 0
            for predictedValidation, groundTruthValidation in zip(validation, Y_validation):
                validationCost += (predictedValidation - groundTruthValidation).dot(predictedValidation - groundTruthValidation) / validation.shape[0]
            print("cost for validation %.10f" % (validationCost))

            testSet = sess.run(pred, feed_dict={x: X_test})
            testCost = 0
            for predictedTest, groundTruthTest in zip(testSet, Y_test):
                testCost += (predictedTest - groundTruthTest).dot(
                    predictedTest - groundTruthTest) / testSet.shape[0]
            print("cost for testSet %.10f" % (testCost))

            if(trainCost>lastTrainCost):
                LR = LR *0.1
                # X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X_train1, Y_train1,test_size=0.125,random_state=epoch+1)
            else:
                hit=0
            if(validationCost>trainCost):
                hit = hit+1
                # X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X_train1, Y_train1,
                #                                                                              test_size=0.125,
                #                                                                              random_state=epoch + 1)
            if(hit >30):
                break;
            lastTrainCost = trainCost

            new_saver.save(sess, '/home/iftimie/PycharmProjects/SirajRavalTutorials/')
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            for i in range(3):
                print ("label value:", batch_y[i],"estimated value:", p[i])



    print ("Optimization Finished!")

    classification = sess.run(pred,feed_dict = {x:X_test})
    listOfTuplesOfFloors = separateFloorsAndPrintAccuracy(X_test,classification,Y_test,'Test Dataset',1.,True)
    plotData(listOfTuplesOfFloors)

    classification = sess.run(pred, feed_dict={x: X_train})
    listOfTuplesOfFloors = separateFloorsAndPrintAccuracy(X_train,classification,Y_train,'Train Dataset',1.,True)
    plotData(listOfTuplesOfFloors)



