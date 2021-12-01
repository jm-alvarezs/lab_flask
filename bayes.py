# Import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
#creating labelEncoder

def getBayes(indice, reaction, correct, error, nechapi):

    features=zip(indice,reaction,correct,error)
    #nechapi=zip(nechapi)

    featuresList=list(features)
    #nechapiList=list(nechapi)

    features=np.array(featuresList)
    #nechapi=np.array(nechapiList)
    
    # Split dataset into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(featuresList, nechapi, test_size=0.4,random_state=34) # 70% training and 30% test

    #Create a Gaussian Classifier
    model = GaussianNB()
    
    # Train the model using the training sets
    model.fit(x_train,y_train)

    #Predict Output
    y_pred=model.predict(featuresList)

    return y_pred