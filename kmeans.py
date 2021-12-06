from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd

def getKmeans(indice, reaction, correct, error):
    #cols=["index","reaccion","correct","error"] #sobre dataset-1
    object = { 'indice': indice, 'reaction': reaction, 'correct': correct, 'error': error }

    data = pd.DataFrame(data=object)
    pca = PCA()
    
    df = pca.fit_transform(data)
    
    kmeans = KMeans(n_clusters= 5)
    
    prediction = kmeans.fit_predict(df)

    #u_labels = np.unique(label)
    #centroids = kmeans.cluster_centers_
       
    return prediction