from ucimlrepo import fetch_ucirepo  # type: ignore
import numpy as np
import random

# fetch dataset 
iris = fetch_ucirepo(id=53)
  
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 

def calc_dist_euclidiana(x, y):
    result = np.sqrt(np.sum((x - y) ** 2))
    return result 

def knn(k):
    # seleccionar valores de las primeras 3 columnas
    values = X.iloc[:, :3]

    # select 100 of the 150 values and delete them from values
    values_total = X.iloc[:, :3]
    training_sample = values_total.sample(100)
    
    # delete from values_total the 100 selected values
    values = values_total.drop(training_sample.index)
    
    coincidencias = 0
    for count, idx in enumerate(values.index):
        distancias = []
        individuo = values.iloc[count]
        clase_ind = y.iloc[idx]
        # calcular distancia euclidiana entre el primer individuo y los 100 seleccionados
        for i in range(len(training_sample)):
            dist = calc_dist_euclidiana(individuo[1], training_sample.iloc[i])
            distancias.append(dist)

        # sort distancias from smallest to largest / may be wrong
        distancias.sort()        
        temp_order_sample = training_sample.iloc[np.argsort(distancias)]

        # select first k distances
        k_distancias = distancias[:k]
        print("Distancias K:", k_distancias)

        # determine classes of the k selected individuals
        clases = []
        for i in range(k):
            index = temp_order_sample.index[i]
            clase = y.iloc[index]
            clases.append(clase.values[0])
        
        # en base a la clase de los k seleccionados determinar la clase del individuo
        resultado = max(set(clases), key = clases.count)
        print("Clase inferida:", resultado, "Clase verdadera:", clase_ind.values[0])
        if resultado == clase_ind.values[0]:
            print("Correcto")
            coincidencias += 1
        else:
            print("Incorrecto")
    print("coincidencias: ", coincidencias, " / 50")

knn(5)
