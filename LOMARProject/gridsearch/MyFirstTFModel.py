# import numpy as np
# import tensorflow as tf
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn import cross_validation
# from sklearn.model_selection import GridSearchCV
#
# sess = tf.Session()
#
# # Wrapper class for the custom kernel chi2_kernel
# class MyTFModel(BaseEstimator,TransformerMixin):
#     def __init__(self,learning_rate = 0.000001):
#         super(MyTFModel,self).__init__()
#         self.learning_rate = learning_rate
#         self.x, self.y, self.new_saver, self.optimizer, self.cost, self.pred = self.createNetwork()
#
#     def createNetwork(self):
#         # Parameters
#         learning_rate = 0.000001
#         dropout_rate = 0.5
#         # Network Parameters
#         n_hidden_1 = 32  # 1st layer number of features
#         n_hidden_2 = 100  # 2nd layer number of features
#         n_hidden_3 = 50
#         n_hidden_4 = 20
#         n_input = 309  # X_train.shape[1]
#         n_classes = 3
#         # tf Graph input
#         x = tf.placeholder("float", [None, 309])
#         y = tf.placeholder("float", [None, 3])
#         np.set_printoptions(precision=2)
#
#         # Create model
#         def multilayer_perceptron(x, weights, biases):
#             # Hidden layers with RELU activation
#             layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
#             layer_1 = tf.nn.relu(layer_1)
#             layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
#             layer_2 = tf.nn.relu(layer_2)
#             layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
#             layer_3 = tf.nn.relu(layer_3)
#             layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
#             layer_4 = tf.nn.relu(layer_4)
#             out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
#             return out_layer
#
#         weights = {
#             'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], 0, 0.1)),
#             'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], 0, 0.1)),
#             'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], 0, 0.1)),
#             'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], 0, 0.1)),
#             'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes], 0, 0.1))
#         }
#         biases = {
#             'b1': tf.Variable(tf.random_normal([n_hidden_1], 0, 0.1)),
#             'b2': tf.Variable(tf.random_normal([n_hidden_2], 0, 0.1)),
#             'b3': tf.Variable(tf.random_normal([n_hidden_3], 0, 0.1)),
#             'b4': tf.Variable(tf.random_normal([n_hidden_4], 0, 0.1)),
#             'out': tf.Variable(tf.random_normal([n_classes], 0, 0.1))
#         }
#         new_saver = tf.train.Saver()
#         pred = multilayer_perceptron(x, weights, biases)
#         cost = tf.reduce_mean(tf.square(pred - y))
#         optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#         return x, y, new_saver, optimizer, cost, pred
#
#     def transform(self, X):
#         classification = sess.run(self.pred, feed_dict={self.x: X})
#         return classification
#
#     def predict(self,X):
#         classification = sess.run(self.pred, feed_dict={self.x: X})
#         return classification
#
#     def fit(self, X, y=None, **fit_params):
#         sess.run(tf.global_variables_initializer())
#         print("running an classifier %.10f"%(self.learning_rate))
#         for epoch in range(100):
#             avg_cost = 0.
#             batch_size = 10
#             total_batch = int(X.shape[0] / batch_size)  # X_train.shape[0] is total lenght (1400 or 490)
#             # Loop over all batches
#             for i in range(total_batch - 1):
#                 batch_x = X[i * batch_size:(i + 1) * batch_size]
#                 batch_y = y[i * batch_size:(i + 1) * batch_size]
#                 # Run optimization op (backprop) and cost op (to get loss value)
#                 _, c, p = sess.run([self.optimizer, self.cost, self.pred], feed_dict={self.x: batch_x, self.y: batch_y})
#                 # Compute average loss
#                 avg_cost += c / total_batch
#
#         return self
#
#     def score(self,X, y, sample_weight=None):
#         testSet = sess.run(self.pred, feed_dict={self.x: X})
#         testCost = 0
#         for predictedTest, groundTruthTest in zip(testSet, y):
#             testCost += (predictedTest - groundTruthTest).dot(
#                 predictedTest - groundTruthTest) / testSet.shape[0]
#         print (testCost)
#
#         return self.compute_accuracy(testSet,y)
#
#     def compute_accuracy(self,Y_pred,Y_true):
#         projectToFloor = True
#         missclasified = 0
#         for classified, groundTruth in zip(Y_pred, Y_true):
#             if (classified[2] < 1.85):
#                 if (projectToFloor):
#                     classified[2] = 0
#             elif (classified[2] < 5.55):
#                 if (projectToFloor):
#                     classified[2] = 3.7
#             elif (classified[2] < 9.25):
#                 if (projectToFloor):
#                     classified[2] = 7.4
#             elif (classified[2] < 20):
#                 if (projectToFloor):
#                     classified[2] = 11.1
#                 # groundTruth[2] = 11.otherCheckpoint
#             if (np.linalg.norm(classified - groundTruth) > 4):
#                 missclasified += 1
#         print("size %d missclasified %d ,..accuracy %.5f" % (Y_pred.shape[0], missclasified,float(Y_pred.shape[0] - missclasified) / float(Y_pred.shape[0])))
#         print ()
#         return float(Y_pred.shape[0] - missclasified) / float(Y_pred.shape[0])
# X_train1 = np.load('X_train1.npy')
# Y_train1 = np.load('Y_train1.npy')
# X_train, X_validation, Y_train,Y_validation = cross_validation.train_test_split(X_train1, Y_train1, test_size=0.125, random_state=236)
#
# param_grid = [{'learning_rate': [0.01]}]
#
#
# scores = ['precision', 'recall']
#
# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print()
#
#     clf = GridSearchCV(MyTFModel(), param_grid,cv=2)
#     #TypeError: If no scoring is specified, the estimator passed should have a 'score' method. The estimator MyTFModel(learning_rate=0.01) does not.
#     clf.fit(X_train, Y_train)
#
#     print("Best parameters set found on development set:")
#     print()
#     print(clf.best_params_)
#     print (clf.best_estimator_)
#     estimator = clf.best_estimator_
#     estimator.__class__ = MyTFModel
#     print (estimator.learning_rate)
#     print()
#     print("Grid scores on development set:")
#
#
#     print("Detailed classification report:")
#     print()
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     print()
#     y_true, y_pred = Y_validation, clf.predict(X_validation)
#     print()