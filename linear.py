import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def getLinear(index, reaction, correct, error, grupo, target):
    object = { 'index': index, 'reaction': reaction, 'correct': correct, 'error': error, 'grupo': grupo, 'target': target }
    data = pd.DataFrame(data=object)
   
    y = data['target']
    x = data.drop('target',axis=1)

    x = np.array(x)
    y = np.array(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, test_size=0.4, random_state = 34)

    LR = LinearRegression()
    LR.fit(x_train,y_train)

    result = np.array(LR.coef_)

    result = np.append(result,LR.intercept_)

    return result




