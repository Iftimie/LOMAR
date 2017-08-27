import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#good_list_of_measurementsX,good_list_of_measurementsY = loadMeasurements()
def loadMeasurements(minStrength):
    mat_contents = sio.loadmat(
        '/home/iftimie/PycharmProjects/SirajRavalTutorials/LOMAR/data/measurements/BUILDING1/UserTrack11ALL_LONG_UniversityBuilding1.mat')
    WLAN_data = mat_contents['user_data_per_measpoint']
    numberOfMeasurements = mat_contents['user_data_per_measpoint'].shape[0]
    good_list_of_measurementsX = []
    good_list_of_measurementsY = []
    for t in range(numberOfMeasurements):
        if (WLAN_data[t][1].shape[1] != 0):
            arrayAPs = WLAN_data[t][1]
            arrayFeature = np.full((1, 309), minStrength)
            for k in range(arrayAPs.shape[1]):  # pt fiecare coloana
                arrayFeature[0, int(arrayAPs[0, k] - 1)] = arrayAPs[1, k]
            good_list_of_measurementsX.append(np.squeeze(np.asarray(arrayFeature)))

            target = WLAN_data[t][0]
            good_list_of_measurementsY.append(np.squeeze(np.asarray(target)))
        else:
            print("empty at %d" % (t + 1))  # actually for example 412 is 413
    return  good_list_of_measurementsX,good_list_of_measurementsY

#XGood,YGood = averageTargets(good_list_of_measurementsX,good_list_of_measurementsY)
def averageTargets(good_list_of_measurementsX,good_list_of_measurementsY):
    mat_contents = sio.loadmat(
        '/home/iftimie/PycharmProjects/SirajRavalTutorials/LOMAR/data/measurements/BUILDING1/UserTrack11ALL_LONG_UniversityBuilding1.mat')
    numberOfMeasurements = mat_contents['user_data_per_measpoint'].shape[0]
    measurementsTargets = []
    good_list_of_measurementsXFiltered = []
    for t in range(numberOfMeasurements):
        if (tuple(good_list_of_measurementsY[t]) in measurementsTargets):
            a = 10
        else:
            measurementsTargets.append(tuple(good_list_of_measurementsY[t]))
            good_list_of_measurementsXFiltered.append(good_list_of_measurementsX[t])

    i = 0
    for obiect in measurementsTargets:
        numberOfDuplicates = 1
        for t in range(numberOfMeasurements):
            if (obiect[0] == tuple(good_list_of_measurementsY[t])[0] and obiect[1] == tuple(good_list_of_measurementsY[t])[1] and obiect[2] ==
                tuple(good_list_of_measurementsY[t])[2]):
                good_list_of_measurementsXFiltered[i] += good_list_of_measurementsX[t]
                numberOfDuplicates += 1
        good_list_of_measurementsXFiltered[i] = np.divide(good_list_of_measurementsXFiltered[i], numberOfDuplicates)
        i += 1

    YGood = []
    for obiect in measurementsTargets:
        YGood.append(obiect)
    YGood = np.array(YGood)
    XGood = np.array(good_list_of_measurementsXFiltered)
    return XGood,YGood

#XGood,YGood = increaseData(XGood,YGood,4,0.2,2)
def increaseData(XGood,YGood,multiplier,sigmaY,sigmaX):
    size = len(YGood)
    for i in range(size):
        for j in range(multiplier):
            new_target = YGood[i] + sigmaY * np.random.randn(1,3)
            new_target[0][2] = YGood[i][2]
            YGood = np.concatenate((YGood, new_target), axis=0)
            new_measurement = XGood[i] + sigmaX * np.random.randn(1,309)
            XGood =np.concatenate((XGood,new_measurement),axis=0)
    return XGood,YGood

#saveData([XGood,YGood],['GeneratedLabels.npy','GeneratedSignals.npy'])
def saveData(listOfArrays,listOfNames):
    for x in range(len(listOfArrays)):
        np.save(listOfNames[x], listOfArrays[x])

def plotData(listOfTuplesOfArrays):
    for x in range(len(listOfTuplesOfArrays)):
        fig = plt.figure('%d' % (x))
        ax = fig.add_subplot(111, projection='3d')
        xs1 = [value[0] for value in listOfTuplesOfArrays[x][0]]# 0 is ground 1 is predicted
        ys1 = [value[1] for value in listOfTuplesOfArrays[x][0]]
        #zs1 = [value[2] for value in listOfTuplesOfArrays[x][0]]
        zs1 = [0 for value in range(len(listOfTuplesOfArrays[x][0]))]#because octavians set does not have multiple floors
        ax.ticklabel_format(useOffset=False)
        ax.scatter(xs1, ys1, zs1, c='b', marker='o')  # c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]

        xs2 = [value[0] for value in listOfTuplesOfArrays[x][1]]
        ys2 = [value[1] for value in listOfTuplesOfArrays[x][1]]
        #zs2 = [value[2] for value in listOfTuplesOfArrays[x][1]]
        zs2 = [0 for value in range(len(listOfTuplesOfArrays[x][0]))] #because octavians set does not have multiple floors
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

def separateFloorsAndPrintAccuracy(XDataSet,classification,label,name,treshold,projectToFloor):
    first_floorG,first_floorP = [],[]
    second_floorG,second_floorP = [],[]
    third_floorG,third_floorP = [],[]
    fourth_floorG,fourth_floorP = [],[]

    missclasified = 0
    missclasifiedFirst = 0
    missclasifiedSecond = 0
    missclasifiedThrid = 0
    missclasifiedFourth = 0

    for classified, groundTruth in zip( classification, label):
        if (classified[2] < 1.85):
            if(projectToFloor):
                classified[2] = 0
            first_floorP.append(np.squeeze(np.asarray(classified)))
            first_floorG.append(np.squeeze(np.asarray(groundTruth)))
            if (np.linalg.norm(classified - groundTruth) > treshold):
                missclasifiedFirst += 1
        elif (classified[2] < 5.55):
            if (projectToFloor):
                classified[2] = 3.7
            second_floorP.append(np.squeeze(np.asarray(classified)))
            second_floorG.append(np.squeeze(np.asarray(groundTruth)))
            if (np.linalg.norm(classified - groundTruth) > treshold):
                missclasifiedSecond += 1
        elif (classified[2] < 9.25):
            if (projectToFloor):
                classified[2] = 7.4
            third_floorP.append(np.squeeze(np.asarray(classified)))
            third_floorG.append(np.squeeze(np.asarray(groundTruth)))
            if (np.linalg.norm(classified - groundTruth) > treshold):
                missclasifiedThrid += 1
        elif (classified[2] < 20):
            if (projectToFloor):
                classified[2] = 11.1
            # groundTruth[2] = 11.otherCheckpoint
            fourth_floorP.append(np.squeeze(np.asarray(classified)))
            fourth_floorG.append(np.squeeze(np.asarray(groundTruth)))
            if (np.linalg.norm(classified - groundTruth) > treshold):
                missclasifiedFourth += 1
        if (np.linalg.norm(classified - groundTruth) > treshold):
            missclasified += 1
    print("%s size %d missclasified %d ,..accuracy %.2f" % (name,
    XDataSet.shape[0], missclasified, float(XDataSet.shape[0] - missclasified) / float(XDataSet.shape[0])))
    print("First size %d missclasified %d ,..accuracy %.2f" % (
    len(first_floorG), missclasifiedFirst, float(len(first_floorG) - missclasifiedFirst) / float(len(first_floorG))))
    print("Second size %d missclasified %d ,..accuracy %.2f" % (len(second_floorG), missclasifiedSecond,
                                                                float(len(second_floorG) - missclasifiedSecond) / float(
                                                                    len(second_floorG))))
    print("Thrid size %d missclasified %d ,..accuracy %.2f" % (
    len(third_floorG), missclasifiedThrid, float(len(third_floorG) - missclasifiedThrid) / float(len(third_floorG))))
    print("Fourth size %d missclasified %d ,..accuracy %.2f\n\n" % (len(fourth_floorG), missclasifiedFourth,
                                                                float(len(fourth_floorG) - missclasifiedFourth) / float(
                                                                    len(fourth_floorG))))
    return [(first_floorG,first_floorP),(second_floorG,second_floorP),(third_floorG,third_floorP),(fourth_floorG,fourth_floorP)]

def plotSet(dataset):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = dataset[:, 0]
    ys = dataset[:, 1]
    zs = dataset[:, 2]
    ax.scatter(xs, ys, zs, c='b', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
