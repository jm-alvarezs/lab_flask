import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def getKnn(indice, reaction, correct, error, nechapi, indicePaciente, reactionPaciente, correctPaciente, errorPaciente):
    #se separan los datos en x y y (variables y resultados)
    object = { 'indice': indice, 'reaction': reaction, 'correct': correct, 'error': error, 'nechapiMayor': nechapi }
    data = pd.DataFrame(data=object)
    x = data.iloc[:,:4].values
    y = data.iloc[:,4].values

    #se separan los datos en x y y (variables y resultados)
    objectPaciente = { 'indice': indicePaciente, 'reaction': reactionPaciente, 'correct': correctPaciente, 'error': errorPaciente}
    dataPaciente = pd.DataFrame(data=objectPaciente)
    xPaciente = dataPaciente.values

    #Se hace un feature scaling para estandarizar datos
    scaler = StandardScaler()
    scaler.fit(x)
    x_scaled = scaler.transform(x)

    #Se hace un feature scaling para estandarizar datos
    scalerPaciente = StandardScaler()
    scalerPaciente.fit(xPaciente)
    x_scaled_paciente = scalerPaciente.transform(xPaciente)

    #Se implementa algoritmo KNN. No existe un numero de vecinos "ideal" pero tras varias pruebas el 10 fue el que dio mejor resultado
    clasificar = KNeighborsClassifier(n_neighbors=10)
    clasificar.fit(x_scaled, y)

    #Se almacenan los datos predichos en el excel de datos
    prediction = clasificar.predict(x_scaled_paciente)
    return prediction


