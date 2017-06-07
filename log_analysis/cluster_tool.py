from sklearn.cluster import KMeans
import time
import numpy as np

def cluster(mat,keys,k=100):
    min_iner=1e10
    min_k=0
    for k in range(5,6):
        #print('using %d centers'%(k))
        estimator=KMeans(init='k-means++',n_clusters=k,n_init=10)
        #print('start k-means')
        time.clock()
        estimator.fit(np.array(mat))
        # print('k-means end, take ',time.clock())
        # print('labels_:',estimator.labels_)
        # print('centers:',estimator.cluster_centers_)
        # print('inertia_:',estimator.inertia_)
        if estimator.inertia_<min_iner:
            min_k=k
            min_iner=estimator.inertia_
            center=estimator.cluster_centers_
            label=estimator.labels_
    print('k=',min_k,'min_iner=',min_iner,'label=',list(label))
    for index in range(min_k):
        print('center ',index)
        for i in range(len(keys)):
            print(keys[i], '-> %.2f'%(center[index][i]))
        print()
    #print('min k, min_iner, label,center:',min_k,min_iner,label,list(center))