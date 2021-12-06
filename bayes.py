import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

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
    
    x_train, x_test, y_train, y_test = train_test_split(featuresList, nechapi, test_size=0.9,random_state=34) # 70% training and 30% test

    model = GaussianNB()
    
    model.fit(x_train,y_train)

    y_pred = model.predict(featuresListPaciente)

    return y_pred