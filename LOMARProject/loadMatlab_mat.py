from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

mat_contents = sio.loadmat('LOMAR/data/measurements/BUILDING1/FingerprintingData_UniversityBuilding1.mat')
WLAN_data = mat_contents['WLAN_data_per_synthpoint']
numberOfAP = mat_contents['WLAN_grid_synthpoint'].shape[1]
numberOfMeasurements = mat_contents['WLAN_data_per_synthpoint'].shape[0]
minStrength  = -100

good_list_of_measurementsX  = []
good_list_of_measurementsY = []
for t in range(numberOfMeasurements):
    if(WLAN_data[t][1].shape[1]!=0): #pt fiecare observatie
        arrayAPs = WLAN_data[t][1]
        arrayFeature = np.full((1, 309), minStrength)
        for k in range(arrayAPs.shape[1]): # pt fiecare coloana
            arrayFeature[0,int(arrayAPs[0,k]-1)] = arrayAPs[1,k]
        good_list_of_measurementsX.append(np.squeeze(np.asarray(arrayFeature)))

        target = WLAN_data[t][0]
        good_list_of_measurementsY.append(np.squeeze(np.asarray(target)))
    else:
        print("empty at %d" % (t + 1))  # actually for example 412 is 413
x = np.array(good_list_of_measurementsX)
y = np.array(good_list_of_measurementsY)


xs = y[:,0]
ys = y[:,1]
zs = y[:,2]
for x in range(727,730):
    print("X = %d" %(x),y[x])
print(zs.shape)

def randrange(n, vmin, vmax):
    return (vmax - vmin)*np.random.rand(n) + vmin

# n=100
# zlow=-50
# zhigh = -25
#
# xs = randrange(n, 23, 32)
# ys = randrange(n, 0, 100)
# zs = randrange(n, zlow, zhigh)
# print (zs.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs, c='r', marker='o') #c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()