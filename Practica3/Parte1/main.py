from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as opt
from sklearn import preprocessing as prep

def dibujarEjemplo(X):
    plt.figure()
    sample = np.random.choice(X.shape[0], 10)
    plt.imshow(X[sample, :].reshape(-1,20).T)
    plt.axis('off')
    plt.show()


def sigmoide(Z):
    return 1 /(1 + np.e**(-Z))

def calcularParteIzq(Y, H):
    return np.dot(np.log(H), Y)

def calcularParteDer(Y, H):
    return np.dot(np.log(1 - H), (1 - Y))

def funcionGradiente(Theta, X, Y):
    H = sigmoide(np.dot(X, Theta))
    return 1 / np.shape(X)[0] * np.dot((H - Y), X)

def funcionCoste(Theta, X, Y):
    H = sigmoide(np.dot(X, Theta))
    return -1 / np.shape(X)[0] * (calcularParteIzq(Y, H) + calcularParteDer(Y, H))

def funcionCosteRegularizado(Theta, X, Y, landa):
    return funcionCoste(Theta, X, Y) + landa/(2*np.shape(X)[0])*(Theta**2).sum()


def funcionGradienteRegularizado(Theta, X, Y, landa):
    return funcionGradiente(Theta, X, Y) + landa/(np.shape(X)[0])*Theta



def problemaRegularizado (X,Y, landa):
    Theta = np.zeros(np.shape(X)[1])
    Y = np.ravel(Y)

    resultRegularizado = opt.fmin_tnc(func = funcionCosteRegularizado, x0 = Theta, fprime = funcionGradienteRegularizado, args = (X, Y, landa), messages=0)
    return resultRegularizado[0]


def oneVsAll( X, y, num_etiquetas, reg):
    matrizThetas = np.empty((num_etiquetas, np.shape(X)[1])) 

    for i in range(num_etiquetas):
        yBooleana = y == i + 1
        matrizThetas[i] = problemaRegularizado(X, yBooleana, reg)

    return matrizThetas
  
    
def calculoProbabilidad(X, Y, matrizThetas):
    contadorAciertos = 0

    for i in range(np.shape(X)[0]):
        probabilidad = sigmoide(np.dot(matrizThetas, X[i]))
        posx = np.where(probabilidad == np.amax(probabilidad))

        if posx[0] + 1 == Y[i] :
           contadorAciertos += 1

    return contadorAciertos / np.shape(X)[0]
       
    



def main():

    data = loadmat('ex3data1.mat')
   
    y = data['y']
    X = data['X']

    X = np.hstack([np.ones([np.shape(X)[0], 1]), X])
    #dibujarEjemplo(X)
    matrizThetas = oneVsAll(X, y, 10, 0.1)
    print(calculoProbabilidad(X, y, matrizThetas))




main()