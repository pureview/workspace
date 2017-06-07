from sklearn.cluster import KMeans
import time
import numpy as np

def cluster(mat,k=3):
    min_iner=1e10
    min_k=0
    for k in range(2,10):
        print('using %d centers'%(k))
        estimator=KMeans(init='k-means++',n_clusters=k,n_init=10)
        print('start k-means')
        time.clock()
        estimator.fit(np.array(mat))
        print('k-means end, take ',time.clock())
        print('labels_:',estimator.labels_)
        print('centers:',estimator.cluster_centers_)
        print('inertia_:',estimator.inertia_)
        if estimator.inertia_<min_iner:
            min_k=k
            min_iner=estimator.inertia_
    print('min k:',min_k)