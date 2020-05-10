from scipy.io import loadmat
import numpy as np
import scipy.optimize as opt
from matplotlib import pyplot as plt
from sklearn.svm import SVC

#APARTADO 1.1 KERNEL LINEAL

def kernelLineal():
    X, y = inicializarBasicos("data/ex6data1.mat")
    #HACEMOS RAVEL PORQUE SI NO DA WARNING AL HACER EL FIT
    y = np.ravel(y)
    svm = SVC( kernel='linear', C=100.0)
    svm.fit(X, y)
    dibujarDatos(X, y, svm)

def inicializarBasicos(archivo):
    valores = loadmat(archivo)
    X, y = valores['X'], valores['y']

    return X, y
    


def dibujarDatos(X, y, svm):
    rangex = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    rangey = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)

    xcomb, ycomb = np.meshgrid(rangex, rangey)

    comb = np.array((np.ravel(xcomb), np.ravel(ycomb)))
    z = svm.predict(np.transpose(comb)).reshape(np.shape(xcomb))
    #RAVEL DE Y YA QUE VIENE EN FORMATO COLUMNA

    pos1 = np.where(y == 1)
    pos2 = np.where(y == 0)
    plt.figure()
    plt.scatter(X[pos1, 0], X[pos1, 1], marker = '+', c = '#FF00FF')
    plt.scatter(X[pos2, 0], X[pos2, 1], marker = 'o', c = '#08E0DE')

    plt.contour(xcomb, ycomb, z, cmap = 'cool')
    plt.show()

#APARTADO 1.2 KERNEL GAUSSIANO
def kernelGaussiano():
    X, y = inicializarBasicos("data/ex6data2.mat")
    
    C = 1
    sigma = 0.1
    #HACEMOS RAVEL PORQUE SI NO DA WARNING AL HACER EL FIT
    y = np.ravel(y)
    svm = SVC( kernel='rbf', C=C, gamma = 1/(2 * sigma ** 2))
    svm.fit(X, y)
    dibujarDatos(X, y, svm)

#APARTADO 1.3 ELECCION DE LOS PARAMETROS

def inicializarValidacion(archivo):
    valores = loadmat(archivo)
    X, y = valores['X'], valores['y']
    Xval, yval = valores['Xval'], valores['yval']

    return X, y, Xval, yval

def eleccionDeParametros():
    X, y, Xval, yval = inicializarValidacion("data/ex6data3.mat")
    conjunto = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    
    #HACEMOS RAVEL PORQUE SI NO DA WARNING AL HACER EL FIT
    y = np.ravel(y)
    yval = np.ravel(yval)
    maxAciertos = 0

    for C in conjunto:
        for sigma in conjunto:
            svm = SVC( kernel='rbf', C=C, gamma = 1/(2 * sigma ** 2))

            svm.fit(X, y)

            ycal = svm.predict(Xval).reshape(np.shape(yval))
            aciertosActuales = np.sum(yval == ycal)

            if aciertosActuales > maxAciertos:
                maxAciertos = aciertosActuales
                maxComparador = svm


    dibujarDatos(X, y, maxComparador)  
            

            



    
   


def main():
    #kernelLineal()
    #kernelGaussiano()
    eleccionDeParametros()
    




main()