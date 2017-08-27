import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from sklearn import cross_validation
import functions
from functions import increaseData
X = np.load('X.npy')
Y = np.load('Y.npy')

# import cv2
# import caffe

# scalerX = preprocessing.StandardScaler()
# X = scalerX.fit_transform(X)w
# scalerY = preprocessing.StandardScaler()
# Y = scalerY.fit_transform(Y)

#X_train, X_test, Y_train,Y_test = cross_validation.train_test_split(X, Y, test_size=0.183673, random_state=236)
#X_train, X_validation,Y_train,Y_validation = cross_validation.train_test_split(X_train, Y_train, test_size=0.125, random_state=236)
from functions import splitTrainValidTest

X_train,Y_train,X_validation,Y_validation,X_test,Y_test = splitTrainValidTest(X,Y)
# X_train, _, Y_train,_ = cross_validation.train_test_split(X_train, Y_train, test_size=0.0000000, random_state=2)


# X_train,Y_train = increaseData(X_train,Y_train,8,0.065,2)
# np.save('Xinc.npy',X_train)
# np.save('Yinc.npy',Y_train)

X_train = np.load('Xinc.npy')
Y_train = np.load('Yinc.npy')


scalerX = preprocessing.StandardScaler()
X_train = scalerX.fit_transform(X_train)
scalerY = preprocessing.StandardScaler()
Y_train = scalerY.fit_transform(Y_train)

X_test = scalerX.transform(X_test)
Y_test = scalerY.transform(Y_test)

X_validation = scalerX.transform(X_validation)
Y_validation = scalerY.transform(Y_validation)
print (len(X_train),len(X_test),len(X_validation))
### [1200,400,40], 0.00006, 0.92, 10
inSize = X.shape[1]
outSize = Y.shape[1]
hiddenSizes = [1200,400,100,15]
with tf.Session() as sess:
    learning_rate_placeholder, x_placeholder, y_placeholder, optimizer, cost, pred,keep_prob = functions.createNetwork(inSize=inSize,outSize=outSize,hiddenSizes=hiddenSizes)

    sess.run(tf.global_variables_initializer())
    new_saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('/media/iftimie/1602BA4902BA2E1D/big_projects/oldProjects/LOMARProject/')
    if ckpt and ckpt.model_checkpoint_path:
        print (ckpt.model_checkpoint_path)
        new_saver.restore(sess, ckpt.model_checkpoint_path)

    lastTrainCost = 9999
    batch_size = 10
    learning_rate_val = 0.000006
    for epoch in range(00000):
        avg_cost = 0.
        total_batch = int(X_train.shape[0] / batch_size)
        # Loop over all batches
        for i in range(total_batch - 1):
            batch_x = X_train[i * batch_size:(i + 1) * batch_size]
            batch_y = Y_train[i * batch_size:(i + 1) * batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, p = sess.run([optimizer, cost, pred], feed_dict={x_placeholder: batch_x, y_placeholder: batch_y,learning_rate_placeholder:learning_rate_val, keep_prob:0.92})
            # Compute average loss
            avg_cost += c / total_batch

        train = sess.run(pred, feed_dict={x_placeholder: X_train,keep_prob:1.0})
        trainCost = 0
        for predictedTrain, groundTruthTrain in zip(train, Y_train):
            trainCost += (predictedTrain - groundTruthTrain).dot(predictedTrain - groundTruthTrain) / train.shape[0]
        print("cost for train %.10f" % (trainCost))
        lastTrainCost = trainCost
        validation = sess.run(pred, feed_dict={x_placeholder: X_validation, keep_prob:1.0})
        validationCost = 0
        for predictedValidation, groundTruthValidation in zip(validation, Y_validation):
            validationCost += (predictedValidation - groundTruthValidation).dot(
                predictedValidation - groundTruthValidation) / validation.shape[0]
        print("cost for validation %.10f" % (validationCost))
        #if (validationCost > 10*trainCost):
        #    break

        if(epoch%20 ==0):
            new_saver.save(sess, '/media/iftimie/1602BA4902BA2E1D/big_projects/oldProjects/LOMARProject/',latest_filename="checkpoint")
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            for i in range(3):
                print("label value:", batch_y[i], "estimated value:", p[i])

    classification = sess.run(pred, feed_dict={x_placeholder: X_test, keep_prob: 1.0})
    classification_real = scalerY.inverse_transform(classification)
    Y_real = scalerY.inverse_transform(Y_test)
    functions.plotData([Y_real, classification_real])
    print("accuracy : ", functions.accuracy(Y_real, classification_real))

    classification = sess.run(pred, feed_dict={x_placeholder: X_train, keep_prob: 1.0})
    classification_real = scalerY.inverse_transform(classification)
    Y_real = scalerY.inverse_transform(Y_train)
    functions.plotData([Y_real, classification_real])
    print("accuracy : ", functions.accuracy(Y_real, classification_real))