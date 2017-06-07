from sklearn import decomposition

def pca(mat):
    pca=decomposition.PCA(5)
    pca.fit(mat)
