import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def getKnn(indice, reaction, correct, error, nechapi):
    #se separan los datos en x y y (variables y resultados)
    object = { 'indice': indice, 'reaction': reaction, 'correct': correct, 'error': error, 'nechapiMayor': nechapi }
    data = pd.DataFrame(data=object)
    x = data.iloc[:,:4].values
    y = data.iloc[:,4].values


    #Se hace un feature scaling para estandarizar datos
    scaler = StandardScaler()
    scaler.fit(x)
    x_scaled = scaler.transform(x)

    #Se implementa algoritmo KNN. No existe un numero de vecinos "ideal" pero tras varias pruebas el 10 fue el que dio mejor resultado
    clasificar = KNeighborsClassifier(n_neighbors=10)
    clasificar.fit(x_scaled, y)

    #Se almacenan los datos predichos en el excel de datos
    data["preditcion"]= clasificar.predict(x_scaled)
    data.to_csv(r'C:/Users/Latitude 7400/Desktop/ITC/PEF/ML/dataset-KNN.csv')


