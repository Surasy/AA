from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as opt
from sklearn import preprocessing as prep


def sigmoide(Z):
    return 1 /(1 + np.e**(-Z))


"""
capaFinal[0] -> Probabilidad de 1
capaFinal[1] -> Probabilidad de 2

capaFinal[9] -> Probabilidad de 0
"""
def comprobarProbabilidad(capaFinal, Y):

    posx = np.argmax(capaFinal, axis= 1)
    Y = np.ravel(Y)
    aciertos = np.sum(posx + 1 == Y)
    
    return aciertos/np.shape(capaFinal)[0]


def tratamientoDeDatos(theta1, theta2, X, Y):
    X = np.hstack([np.ones([np.shape(X)[0], 1]), X])

    capaIntermedia = sigmoide(np.dot(X, np.transpose(theta1)))

    capaIntermedia = np.hstack([np.ones([np.shape(capaIntermedia)[0], 1]), capaIntermedia])
    capaFinal = sigmoide(np.dot(capaIntermedia, np.transpose(theta2)))
    
    print(comprobarProbabilidad(capaFinal, Y))

    



def cargaCasos(nombre):

    data = loadmat(nombre)
   
    y = data['y']
    X = data['X']

    return X, y


def cargarDatos(nombre):

    weights = loadmat(nombre)
    theta1, theta2 = weights['Theta1'], weights['Theta2']
    # Theta1 25 x 401
    # Theta2 10 x 26
    return theta1, theta2


def main():

    theta1, theta2 = cargarDatos('ex3weights.mat')
   
    X, y = cargaCasos('ex3data1.mat')

    tratamientoDeDatos(theta1, theta2, X, y)
   
  



main()