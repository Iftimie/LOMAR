import numpy as np
from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering
from sklearn.decomposition import PCA
from scipy.io import loadmat
import time

def data(folder, key, ap_count=None):
    mat = loadmat(folder)

    if not ap_count:
        ap_count = mat['WLAN_grid_synthpoint'].shape[1]
    
    X = []
    y = []
    for point in mat[key]:
        if point[1].size:
            fingerprint = np.zeros(ap_count)
            ap = point[1][0].astype(np.int) - 1
            fingerprint[ap] = point[1][1]

            X.append(fingerprint)
            y.append(point[0][0])
    
    X = np.array(X)
    y = np.array(y)

    X[X == 0] = np.nan

    return X, y

def bayes_position(X_train, y_train, X_test, N, sigma, eps, th, lth, div, y_test):
    diff = X_train - X_test

    proba = 1/(np.sqrt(2*np.pi)*sigma)*np.exp( \
        -np.power(diff, 2)/(2.0*sigma**2))

    proba[np.isnan(proba)] = eps
    proba[proba < th] = eps
    proba = np.log(proba)
    cost = np.sum(proba, axis=1)

    inv = np.zeros(X_train.shape[0])
    for i in range(X_train.shape[0]):
        a = np.logical_and(~np.isnan(X_train[i]), np.isnan(X_test))
        b = np.logical_and(np.isnan(X_train[i]), ~np.isnan(X_test))

        nfound = np.concatenate((X_train[i,a], X_test[b]))
        for v in nfound[nfound > lth]:
            inv[i] += v - lth
            
    inv /= div
    cost -= inv

    idx = np.argsort(cost)[::-1]

    bias = 3
    position = np.zeros(3)
    for i in range(N):
        weight = N-i
        if i == 0:
            weight += bias

        position += weight*y_train[idx[i]]

    position /= N*(N+1)/2+bias

    return (np.array(position), np.mean(inv[idx[:20]]))

def distance(a, b):
    return np.sqrt(np.sum(np.power(a-b, 2)))

def cluster_subset(clusters, labels, pos, X_test):
    d = []
    for i,c in enumerate(kmeans.cluster_centers_):
        d.append(distance(pos[:2], c[:2]))

    center = np.argmin(d)

    return (ss[center], cs[center])

def position_route(X_train, y_train, X_test, y_test, clusters, labels,
                   N=5, sigma=5, eps=3e-4, th=1e-25, lth=-85, div=10):

    error = []
    fdetect = 0
    y_pred = []
    cused = []

    for i in range(X_test.shape[0]):
        if i > 1:
            subset, c = cluster_subset(clusters, labels, pos, X_test[i])
            cused.append(c)
        else:
            subset = np.ones(X_train.shape[0]).astype(np.bool)

        pos, q = bayes_position(X_train[subset], y_train[subset], X_test[i], N, sigma, 
                                eps, th, lth, div, y_test[i])

        if q > 50:
            pos, _ = bayes_position(X_train, y_train, X_test[i], N, sigma, 
                                    eps, th, lth, div, y_test[i])
        
        pos[2] = floors[np.argmin(np.abs(floors-pos[2]))]

        if i > 1:
            y_pred.append(pos)
            error.append(distance(y_test[i], y_pred[-1]))
            fdetect += y_pred[-1][2] == y_test[i][2]
    
    return (np.array(y_pred), np.array(error), fdetect, np.array(cused))

# training data
TRAIN_KEY = 'WLAN_data_per_synthpoint'
TEST_KEY = 'user_data_per_measpoint'

X_train, y_train = data('2/train.mat', TRAIN_KEY)

ap_count = X_train.shape[1]
floors = np.unique(y_train[:,2])

# test data
X_routes = []
y_routes = []

for route in range(0,7):
    test_file = '2/test_'+str(route+1)+'.mat'
    X, y = data(test_file, TEST_KEY, ap_count)
    X_routes.append(X)
    y_routes.append(y)

error = np.array([])
cused = np.array([])
fdetect = 0
tsum = 0

for route in range(0,7):
    for repeat in range(1):
        X_ktrain = X_train.copy()
        y_ktrain = y_train.copy()
        for i in range(0,7):
            if i != route:
                X_ktrain = np.concatenate((X_ktrain, X_routes[i]))
                y_ktrain = np.concatenate((y_ktrain, y_routes[i]))

        X_test = X_routes[route]
        y_test = y_routes[route]

        X_aux = X_ktrain.copy()
        X_aux[np.isnan(X_aux)] = 0

        M = X_ktrain.shape[1]
        corr = np.zeros((M,M))
        cth = 500
        keep = np.ones(M).astype(np.bool)
        for i in range(M):
            for j in range(i,M):
                if i != j:
                    diff = np.abs(X_aux[:,i] - X_aux[:,j])
                    corr[i,j] = corr[j,i] = np.sum(diff)
                else:
                    corr[i,j] = cth

            if keep[i] and np.sum(corr[i,:] < cth) > 0:
                for p in np.where(corr[i,:] < cth)[0]:
                    keep[p] = False

        X_ktrain = X_ktrain[:,keep]
        X_test = X_test[:,keep]

        C = 25
    
        kmeans = KMeans(n_clusters=C, n_init=500, n_jobs=2, tol=1e-9)
        labels = kmeans.fit_predict(y_ktrain)
        clusters = kmeans.cluster_centers_

        N = X_ktrain.shape[0]
        aux = np.zeros((C,C))
        for i in range(N):
            dist = np.zeros(N)
            for j in range(N):
                dist[j] = distance(y_ktrain[i], y_ktrain[j])
        
            idx = np.argsort(dist) 
        
            for p in np.where(labels[idx] != labels[i])[0]:
                if dist[idx[p]] < 10:
                    aux[labels[i],labels[idx[p]]] += 1    
    
        ss = np.zeros((C,labels.size)).astype(np.bool)
        cs = np.zeros(C)
        rssl = []
        rssc = []
        for c in range(C):
            aux[c,c] = 1

            for i in np.where(aux[c] != 0)[0]:
                ss[c] = np.logical_or(ss[c], labels == i)
                cs[c] += 1

        t = time.clock()

        y, e, f, c = position_route(X_ktrain, y_ktrain, X_test, y_test, clusters, 
                                    labels, N=5, eps=1e-3)

        tsum += time.clock() - t

        error = np.concatenate((error, e))
        cused = np.concatenate((cused, c))
        fdetect += f
    
        print '%d: %.2lf' % (route, np.mean(e))

print 'floor %.3lf' % (float(fdetect) / error.shape[0])
print 'total %.2lf' % np.mean(error)

if cused.size > 0:
    print 'cused %.2lf' % np.mean(cused)

print 'time  %.2lf' % tsum

np.save('error.npy', error)
