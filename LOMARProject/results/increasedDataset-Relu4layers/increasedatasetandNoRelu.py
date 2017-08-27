from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from sklearn import cross_validation
from usefulFunctions.functions import loadMeasurements,averageTargets,increaseData,saveData,plotData
from usefulFunctions.functions import separateFloorsAndPrintAccuracy
from usefulFunctions.networkFunctions import createNetwork

# y = np.load('GeneratedLabels.npy')
# x = np.load('GeneratedSignals.npy')

good_list_of_measurementsX,good_list_of_measurementsY = loadMeasurements(minStrength = -120)
XGood,YGood = averageTargets(good_list_of_measurementsX,good_list_of_measurementsY)

X_train1, X_test, Y_train1, Y_test = cross_validation.train_test_split(XGood, YGood, test_size=0.183673, random_state=12)

#X_train1,Y_train1 = increaseData(X_train1,Y_train1,4,0.2,2)
#saveData([X_train1,Y_train1],['X_train1.npy','Y_train1.npy'])
X_train1 = np.load('X_train1.npy')
Y_train1 = np.load('Y_train1.npy')
X_train, X_validation, Y_train,Y_validation = cross_validation.train_test_split(X_train1, Y_train1, test_size=0.125, random_state=236)



x,y,new_saver,optimizer,cost,pred = createNetwork()
hit = 0

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
            _, c, p = sess.run([optimizer, cost, pred], feed_dict={x: batch_x, y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch

        if epoch % 20 == 0:
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

            if(validationCost>trainCost):
                hit+=1
                X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X_train1, Y_train1,test_size=0.125,random_state=epoch+1)

            # else:
            #     hit=0
            # if(hit >20):
            #     break;

            new_saver.save(sess, '/home/iftimie/PycharmProjects/SirajRavalTutorials/')
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            for i in range(3):
                print ("label value:", batch_y[i],"estimated value:", p[i])



    print ("Optimization Finished!")

    classification = sess.run(pred,feed_dict = {x:X_test})
    listOfTuplesOfFloors = separateFloorsAndPrintAccuracy(X_test,classification,Y_test,'Test Dataset',1.,False)
    plotData(listOfTuplesOfFloors)

    classification = sess.run(pred, feed_dict={x: X_train})
    listOfTuplesOfFloors = separateFloorsAndPrintAccuracy(X_train,classification,Y_train,'Train Dataset',1.,False)
    plotData(listOfTuplesOfFloors)



