#Importing required modules
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd

#Load Data

def getKmeans(indice, reaction, correct, error):
    #cols=["index","reaccion","correct","error"] #sobre dataset-1
    object = { 'indice': indice, 'reaction': reaction, 'correct': correct, 'error': error }
    data = pd.DataFrame(data=object)
    pca = PCA()
    
    #Transform the data
    df = pca.fit_transform(data)
    
    #Initialize the class object
    kmeans = KMeans(n_clusters= 5)
    
    #predict the labels of clusters.
    prediction = kmeans.fit_predict(df)

    #Getting unique labels
    #u_labels = np.unique(label)
    centroids = kmeans.cluster_centers_
       
    return prediction, centroids