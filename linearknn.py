import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def getLinearKnn(anger, sensation, emotional, sociablity, motivation, col):
    object = { 'anger': anger, 'sensation': sensation, 'emotional': emotional, 'sociability': sociablity, 'motivation': motivation }
    data = pd.DataFrame(data=object)    
    y = data[col]
    x = data.drop(col,axis=1)

    x=np.array(x)
    y=np.array(y) 
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, test_size = 0.4, random_state = 34)

    LR = LinearRegression()
    LR.fit(x_train,y_train)

    y_prediction =  LR.predict(x_test)
    return y_prediction
