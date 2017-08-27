import numpy as np

file = open("logfile.txt","r")

setMACS = set()
for line in file:
    linedata = line.split(",")
    for x in range(2,len(linedata),2):
        setMACS.add(linedata[x])
file.close()

print (setMACS)
listaMACS = list(setMACS)
listaMACS.sort()
print(listaMACS)
if 'C8:3A:35::98:50' in listaMACS:
    print ("este")
else:
    print ("nu este")

print (len(listaMACS))
print (listaMACS.index('64:66:B3:45:13:46'))

inputSize = len(listaMACS)
minStrength = -100

Y = []
X = []
file = open("logfile.txt","r")
for line in file:
    linedata = line.split(",")
    Y.append([linedata[0],linedata[1]])
    arrayFeature = np.full((1, inputSize), minStrength)
    for x in range(2,len(linedata),2):
        index = listaMACS.index(linedata[x])
        arrayFeature[0,index] = float(linedata[x+1])
    X.append(np.squeeze(np.asarray(arrayFeature)))

print (np.array(Y))
print (np.array(X))


