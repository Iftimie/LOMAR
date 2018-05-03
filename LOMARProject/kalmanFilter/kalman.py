import numpy as np
from numpy.linalg import inv
import cv2
from munkres import Munkres
from scipy.optimize import linear_sum_assignment

class KalmanFilter:

    dt = 1
    processNoise =0.1
    u = 0

    def __init__(self,tkn_x = 100,tkn_y= 100, maxTrackLength = 20):
        self.tkn_x = tkn_x
        self.tkn_y = tkn_y
        self.maxTrackLength = maxTrackLength
        self.Ez = np.array( [[self.tkn_x,0],[0,self.tkn_y]]) #error in measurement
        self.Ex = np.array([[self.dt**4/4 , 0,           self.dt**3/2 ,0           ],
                            [0,             self.dt**4/4,0,            self.dt**3/2],
                            [self.dt**3/2,  0,           self.dt**2,   0           ],
                            [0,             self.dt**3/2,0             ,self.dt**2]])#error in prediction/ covariance matrix
        self.Ex = self.Ex * self.processNoise**2
        self.P = self.Ex

        self.A = np.array([[1, 0, self.dt, 0      ],
                           [0, 1, 0,       self.dt],
                           [0, 0, 1,       0      ],
                           [0, 0, 0,       1      ]])
        self.B = np.array([[self.dt**2/2],
                           [self.dt**2/2],
                           [self.dt     ],
                           [self.dt     ]])
        self.C = np.array([[1,0,0,0],
                           [0,1,0,0]])

        self.K = None
        self.trackedStates = []
        self.tracks = []
        self.speedVectors = [] ##### speedVectors update


    #vector<pair<int, arma::Mat<double>>> trackedStates
    #this->trackedStates[i].second
    def predict(self):
        for i in range(len(self.trackedStates)):
            self.trackedStates[i][1] = self.A.dot(self.trackedStates[i][1])+self.B*self.u

        self.P = (self.A.dot(self.P)).dot(self.A.T) + self.Ex
        self.K = (self.P.dot(self.C.T)).dot( inv((self.C.dot( self.P)).dot( self.C.T) +self.Ez))

    def getSpeedVectorsAndStatePoint(self):
        statePoints = []
        speedVectors = []
        for i in range(len(self.trackedStates)):
            state = np.array([self.trackedStates[i][1][0, 0], self.trackedStates[i][1][1, 0]])
            speedVector = np.zeros((1,2))
            for j in range(len(self.speedVectors[i])):
                speedVector += self.speedVectors[i][j]
            speedVector /= len(self.speedVectors[i])
            speedVector = np.nan_to_num(speedVector)
            statePoints.append(state)
            speedVectors.append(speedVector)
        return statePoints,speedVectors

    def draw(self,img,detections):
        for i in range(len(detections)):
            img = cv2.circle(img,detections[i],3,(255,255,255),2)
        for i in range(len(self.trackedStates)):
            p = (int(self.trackedStates[i][1][0,0]),int(self.trackedStates[i][1][1,0]))
            img = cv2.circle(img,p,3,(40,190,70),2)
        for i in range(len(self.tracks)):
            for j in range(len(self.tracks[i])-1):
                color = ((i*20)%255,255-(i*20)%255,(i*44)%255)
                img = cv2.line(img,self.tracks[i][j],self.tracks[i][j+1],color,2)

        return img

    def getPoint(self):
        return [self.trackedStates[0][1][0,0],self.trackedStates[0][1][1,0]]

    def createCostMatrix(self,detections):
        m = len(self.trackedStates)
        n = len(detections)
        dummy_columns = m-n
        if dummy_columns<0:
            dummy_columns = 0
        if n<m:
            n=m
        costMatrix = np.zeros((m, n))
        for i in range(m):
            trackedPoint = np.array([self.trackedStates[i][1][0,0],self.trackedStates[i][1][1,0]])
            for j in range(n-dummy_columns):
                costMatrix[i,j] = int(np.linalg.norm(trackedPoint-detections[j]))
                if costMatrix[i,j]==0:
                    costMatrix[i,j] =1

        costMatrix = costMatrix.astype(int)

        for i in range(m):
            for j in range(n-dummy_columns,n):
                costMatrix[i,j]=0

        return costMatrix,m,n,dummy_columns

    def getTrackingsWithoutDetectionsAndIgnoredDetections(self,assignements,m,n,dummy_columns):
        assignementMatrix = np.zeros((m,n))
        for row,column in assignements:
            assignementMatrix[row,column] = 1
        trackingsWithoutDetections = []
        for i in range(m):
            zeros = 0
            for j in range(n-dummy_columns):
                if assignementMatrix[i,j]==0:
                    zeros+=1
            if zeros == n-dummy_columns:
                trackingsWithoutDetections.append(i)

        ignoredDetections = []
        for j in range(n):
            zeros = 0
            for i in range(m):
                if assignementMatrix[i,j]==0:
                    zeros+=1
            if zeros == m:
                ignoredDetections.append(j)
        return trackingsWithoutDetections,ignoredDetections




    def update(self,detections):
        costMatrix,m,n,dummy_columns = self.createCostMatrix(detections)
        munkres = Munkres()

        assignements = []
        if m!=0 and n!=0:
            assignements = munkres.compute(costMatrix)


        trackingsWithoutDetections,ignoredDetections = self.getTrackingsWithoutDetectionsAndIgnoredDetections(assignements,m,n,dummy_columns)

        # filter the assignements for the dummy_columns
        assignements = [assignement for assignement in assignements if not assignement[1] >= n - dummy_columns]

        for i in range(len(assignements)):
            row = assignements[i][0]
            column = assignements[i][1]
            detectedPoint = np.array([[detections[column][0]],  # x
                                      [detections[column][1]]])  # y
            self.trackedStates[row][1] = self.trackedStates[row][1] + self.K.dot (
            detectedPoint - self.C.dot( self.trackedStates[row][1]))
            point = (int(self.trackedStates[row][1][0, 0]), int(self.trackedStates[row][1][1, 0]))

            speedVector = np.array([self.trackedStates[row][1][2,0], self.trackedStates[row][1][3,0]]) ##### speedVectors update

            self.speedVectors[row].append(speedVector) ##### speedVectors update
            self.tracks[row].append(point)

        I = np.identity(4)
        self.P = (I - self.K.dot(self.C)).dot(self.P)

        for i in range(len(trackingsWithoutDetections)):
            row = trackingsWithoutDetections[i]
            self.trackedStates[row][0] +=1

        #limit track length
        for i in range(len(self.tracks)):
            if len(self.tracks[i]) > self.maxTrackLength:
                del self.tracks[i][0]
                del self.speedVectors[i][0] ##### speedVectors update

        #erase trackedStates that did not get assignmenets for 6 times
        self.trackedStates = [state for state in self.trackedStates if not self.determine(state)]
        self.tracks = [track for state,track in zip(self.trackedStates,self.tracks) if not self.determine(state)]
        self.speedVectors = [speedVector for state,speedVector in zip(self.trackedStates,self.speedVectors) if not self.determine(state)]

        #find the new detection
        for i in range(len(ignoredDetections)):
            col = ignoredDetections[i]
            detectedPoint = np.array([[detections[col][0]],
                                      [detections[col][1]],
                                      [0                 ],
                                      [0                 ]])
            trackedState = [0,detectedPoint]
            self.trackedStates.append(trackedState)
            self.tracks.append([detections[col]])
            self.speedVectors.append([])



    def determine(self,state):
        if state[0] > 6:
            return True
        else:
            return False