import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Qt5Agg')

if __name__ == '__main__':
    x = []
    x1 = []
    y0 = []
    y1 = []
    y2 = []

    fig, axs = plt.subplots(2)

    for i in range(10):
        print("h")
        x.append(i)

        y0.append(i * 2)
        if i % 2 == 0:
            x1.append(i)
            y1.append(i**2)
        y2.append([i*2, i*3])

        plt.scatter(x1, y1)

        plt.scatter(x, y0)
        plt.draw()


        plt.pause(0.001)

    plt.show()