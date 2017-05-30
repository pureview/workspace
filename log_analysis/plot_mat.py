import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot(mat,names):
    ax=plt.subplot(111,projection='3d')
    mat=np.array(mat)
    ax.scatter(mat[:,0],mat[:,1],mat[:,2])
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_zlabel(names[2])
    plt.show()