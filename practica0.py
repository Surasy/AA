# Practica realizada por Daniel Padilla y Sofia Prieto
import time
import numpy as np
from matplotlib import pyplot as plt

#Algoritmo aplicando operaciones de vectores
def integra_mc_rapido(fun, a, b, num_puntos=10000):
    #Guardamos la marca inicial
    tic = time.process_time()

    #Generamos num_puntos puntos entre el rango de a y b
    x = np.linspace(a, b, num_puntos)

    #Hallamos su y equivalente
    y = fun(x)
    maxval = np.amax(y)

    #Generamos num_puntos puntos aleatorios como máximo el valor maximo
    numsrands = np.random.rand(num_puntos) * maxval

    #Hallamos el numero de puntos por debajo de la grafica
    mask = numsrands <= y
    debajo = np.sum(mask)

    #Obtenemos el valor del area aproximado
    area = debajo/num_puntos*(b - a)*maxval

    #Guardamos la marca final y sacamos la diferencia
    toc = time.process_time()
    return 1000 * (toc - tic)


#Algoritmo aplicano operaciones iterativas
def integra_mc_lento(fun, a, b, num_puntos=10000):
    #Guardamos la marca inicial
    tic = time.process_time()
    #Generamos num_puntos puntos entre el rango de a y b
    x = np.linspace(a, b, num_puntos)
    #Hallamos su y equivalente
    y = [fun(num) for num in x]
    maxval = np.amax(y)

    #Generamos num_puntos puntos aleatorios como máximo el valor maximo
    numsrands = np.random.rand(num_puntos) * maxval

    #Hallamos el numero de puntos por debajo de la grafica
    debajo = 0
    for i in range(num_puntos):
        debajo += numsrands[i] <= y[i]

    #Obtenemos el valor del area aproximado        
    area = debajo/num_puntos*(b - a)*maxval

    #Guardamos la marca final y sacamos la diferencia  
    toc = time.process_time()
    return 1000 * (toc - tic)




def main():
    #Generamos el 20 tamaños de puntos entre 100 y 1000000
    sizes = np.linspace(100, 1000000, 20)
    times_lento = []
    times_rapido = []
    #Ejecutamos el tamaño de puntos en ambo algoritmos
    contador = 0
    for size in sizes:
        times_lento += [integra_mc_lento(np.sin, 0, np.pi, int(size))]
        times_rapido += [integra_mc_rapido(np.sin, 0, np.pi, int(size))]
        contador += 1
        print(contador)


    #Generamos la grafica para comparar resultados
    plt.figure()
    plt.scatter(sizes, times_lento, c='#75b79e', label='iterativo')
    plt.scatter(sizes, times_rapido, c='#6a8caf', label='vectorizacion')
    plt.legend()
    plt.show()


main()