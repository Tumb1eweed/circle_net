import pyransac3d as pyrsc
import numpy as np

if __name__ == '__main__':
    data = np.loadtxt('./circledata.txt')[:,:3]
    sph = pyrsc.Circle()
    center, axis, radius, inliers = sph.fit(data, thresh=0.002, maxIteration=100)
    print(center)
    print(axis)
    print(radius)