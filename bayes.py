# Import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
#creating labelEncoder

def getBayes(index, reaction, correct, error, nechapi, indexPaciente, reactionPaciente, correctPaciente, errorPaciente):

    features = zip(index,reaction,correct,error)
    featuresPaciente = zip(indexPaciente, reactionPaciente, correctPaciente, errorPaciente)
    #nechapi=zip(nechapi)

    featuresList = list(features)
    featuresListPaciente = list(featuresPaciente)
    #nechapiList=list(nechapi)

    features = np.array(featuresList)
    featuresPaciente = np.array(featuresListPaciente)
    #nechapi=np.array(nechapiList)
    
    # Split dataset into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(featuresList, nechapi, test_size=0.9,random_state=34) # 70% training and 30% test

    #Create a Gaussian Classifier
    model = GaussianNB()
    
    # Train the model using the training sets
    model.fit(x_train,y_train)

    #Predict Output
    y_pred = model.predict(featuresListPaciente)

    return y_pred