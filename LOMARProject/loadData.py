import numpy as np
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#creating the set of macs
file = open("dataset.txt","r")
setMACS = set()
for line in file:
    linedata = line.split(",")
    for x in range(2,len(linedata)):
        rssi = linedata[x].split(" ")[0]
        mac = linedata[x].split(" ")[1].replace("\n","")
        # some mac strings had at the end \n resulting in duplicates like 'E8:94:F6:67:A5:EB' and 'E8:94:F6:67:A5:EB\n'
        setMACS.add(mac)
file.close()


listaMACS = list(setMACS)
listaMACS.sort()
#saving the distinct MACs
print ("number of MACS",len(listaMACS))
# file = open("distinctMACs.txt","w")
# for mac in listaMACS:
#     file.write(mac+"\n")
# file.close()

#creating the training data
inputSize = len(listaMACS)
minStrength = 0
Y = []
X = []
file = open("dataset.txt","r")
for line in file:
    linedata = line.split(",")
    Y.append([float(linedata[0]),float(linedata[1])]) #x and y coordinates
    featureVector = np.full((1, inputSize), minStrength)
    for x in range(2, len(linedata)):
        rssi = linedata[x].split(" ")[0]
        mac = linedata[x].split(" ")[1].replace("\n", "")
        index = listaMACS.index(mac)
        featureVector[0, index] = float(rssi)
    X.append(np.squeeze(np.asarray(featureVector)))



X = np.array(X)
Y = np.array(Y)
print (X.shape)
np.save("X.npy",X)
np.save("Y.npy",Y)

file = open("LOMAR_DATASET.csv","w")
for i in range(0,X.shape[1]):
    file.write(listaMACS[i]+",")
file.write("x,y\n")
for i in range(0,len(Y)):
    for j in range(0, X.shape[1]):
        file.write(str(X[i,j])+",")
    file.write(str(Y[i, 0]) + "," + str(Y[i, 1])+"\n")
file.close()

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


plotSet(Y)