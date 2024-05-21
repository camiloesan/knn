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

    # seleccionar 100 valores random de los 150 valores totales
    values_total = X.iloc[:, :3]
    random_indices = np.random.choice(X.index, size=100, replace=False)
    training_sample = values_total.loc[random_indices]
    
    # eliminar valores seleccionados
    values = values_total.drop(training_sample.index)
    
    coincidencias = 0
    for count, idx in enumerate(values.index):
        distancias = []
        individuo = values.iloc[count]
        print("count", count, "idx:", idx)
        clase_ind = y.iloc[idx]
        # calcular distancia euclidiana entre el primer individuo y los 100 seleccionados
        for i in range(len(training_sample)):
            dist = calc_dist_euclidiana(individuo, training_sample.iloc[i])
            distancias.append(dist)

        # ordenar distancias de menor a mayor
        temp_order_sample = training_sample.iloc[np.argsort(distancias)]
        distancias.sort()        

        # seleccionar las primeras k distancias
        k_distancias = distancias[:k]
        print("Distancias K:", k_distancias)

        # determinar la clase de los k individuos
        clases = []
        for i in range(k):
            index = temp_order_sample.index[i]
            clase = y.iloc[index]
            clases.append(clase.values[0])
        
        # en base a la clase de los k seleccionados determinar la clase del individuo
        resultado = max(set(clases), key = clases.count)
        print("Clase inferida:", resultado, "Clase verdadera:", clase_ind.values[0])
        
        # aumentar contador de coincidencias si la clase inferida es igual a la clase verdadera
        if resultado == clase_ind.values[0]:
            print("Correcto")
            coincidencias += 1
        else:
            print("Incorrecto")
        print()
        
    print("coincidencias: ", coincidencias, " / 50")
    print("Porcentaje de acierto: ", coincidencias / 50 * 100, "%")

knn(10)
