from ucimlrepo import fetch_ucirepo  # type: ignore
import numpy as np
import random

# fetch dataset 
iris = fetch_ucirepo(id=53)
  
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 

def calc_dist_euclidiana(x, y):
    print ("x:", x, "y:", y)
    return np.sqrt(np.sum((x - y) ** 2))

def knn(k):
    # seleccionar valores de la primera columna
    values = X.iloc[:, 0]

    # select 100 of the 150 values and delete them from values
    s = random.sample(list(enumerate(values)), 100)
    indices_to_delete = [index for index, _ in s]
    values_idx = list(enumerate(values))
    values_idx = [x for x in values_idx if x[0] not in indices_to_delete]
    values = values.drop(indices_to_delete)
    
    # calcular distancia euclidiana entre el primer individuo y los 100 seleccionados
    distancias = []
    random_index = random.randint(0, 50) # en lugar de random, hacer por cada individuo
    individuo = values_idx[random_index]
    clase_ind = y.iloc[individuo[0]]
    print("Individuo:", individuo)
    for i in range(len(s)):
        dist = calc_dist_euclidiana(individuo[1], s[i][1])
        distancias.append(dist)
        
    # sort distancias from smallest to largest
    distancias.sort()

    # apply the same sort criteria to s
    s.sort(key=lambda x: calc_dist_euclidiana(individuo[1], x[1]))

    # select first 5 distances
    k_distancias = distancias[:k]
    print("Distancias:", distancias)
    print("Distancias K:", k_distancias)

    # determine classes of the 5 selected individuals
    clases = []
    for i in range(k):
        index = s[i][0]
        clase = y.iloc[index]
        clases.append(clase)
    print("Clases:\n", clases)
    
    # en base a la clase de los k seleccionados determinar la clase del individuo
    resultado = np.max(clases)
    print("Clase verdadera:", clase_ind.values[0])
    print("Clase inferida:", resultado)

knn(5)
