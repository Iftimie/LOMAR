from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn import cross_validation
import functions
from sklearn import preprocessing


X = np.load('X.npy')
Y = np.load('Y.npy')

scalerX = preprocessing.StandardScaler()
X = scalerX.fit_transform(X)
scalerY = preprocessing.StandardScaler()
Y = scalerY.fit_transform(Y)

X_train, X_test, Y_train,Y_test = cross_validation.train_test_split(X, Y, test_size=0.183673, random_state=236)

clf = MLPRegressor(solver='lbfgs', alpha=1e-7, hidden_layer_sizes=(120,50,10), random_state=865,learning_rate_init=0.00001, max_iter=200,activation =('tanh'))
clf.fit(X_train,Y_train)

y_pred = clf.predict(X_test)

y_pred = scalerY.inverse_transform(y_pred)
Y_test = scalerY.inverse_transform(Y_test)

functions.plotData([Y_test,y_pred])
print (functions.accuracy(Y_test,y_pred))