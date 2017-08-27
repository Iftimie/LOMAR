import matplotlib.pyplot as plt
import numpy as np
plt.axis([0, 40, 0, 4])
plt.ion()


for i in range(100):
    y = np.random.random()
    x = np.random.random()
    plt.scatter(x, y, color="yellow", edgecolors="black")
    plt.pause(0.005)

    if i % 20:
        plt.clf()
        plt.axis([0, 40, 0, 4])
        plt.ion()

while True:
    plt.pause(0.005)