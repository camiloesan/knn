from ucimlrepo import fetch_ucirepo  # type: ignore
import numpy as np
import random

# fetch dataset 
iris = fetch_ucirepo(id=53)
  
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 

def calc_dist_euclidiana(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def knn(k):
    # seleccionar valores de la primera columna
    values = X.iloc[:, 0]

    # select 100 of the 150 values and delete them from values
    s = random.sample(list(enumerate(values)), 100)
    indices_to_delete = [index for index, _ in s]
    values = values.drop(indices_to_delete)

    # calcular distancia euclidiana entre el primer individuo y los 100 seleccionados
    distancias = []
    random_index = random.randint(0, 50) # en lugar de random, hacer por cada individuo
    individuo = values.iloc[random_index]
    for i in range(len(s)):
        dist = calc_dist_euclidiana(individuo, s[i])
        distancias.append(dist)
        
    # sort distancias from smallest to largest
    distancias.sort()

    # apply the same sort criteria to s
    s.sort(key=lambda x: calc_dist_euclidiana(individuo, x[1]))

    # select first 5 distances
    k_distancias = distancias[:k]
    print("Distancias K:", k_distancias)

    # determine classes of the 5 selected individuals
    clases = []
    for i in range(k):
        index = s[i][0]
        clase = y.iloc[index]
        clases.append(clase)
    print("Clases:\n", clases)
    # en base a la clase de los 5 seleccionados determinar la clase del individuo
    # (votacion)
    # aplicar peso a las distancias

knn(5)
