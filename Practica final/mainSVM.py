from pandas.io.parsers import read_csv
import numpy as np
import scipy.optimize as opt
from sklearn.svm import SVC
import math 


def carga_csv(file_name):
    # S -> plantarse 0, H -> pedir carta 1
    # W -> ganar 1, L -> perder 0
    valores = read_csv(file_name, header=0).values
    valores[:, 4] = (valores[:, 4] == "H")
    valores[:, 5] = (valores[:, 5] == "W")

    return valores.astype(float)


def fraccionar(X, Y, porcentajeTrain, porcentajeVal, porcentajeTest):

    #total = len(X)
    total = 20000

    indiceTrain = math.floor(total * porcentajeTrain/100)
    indiceVal = math.floor(total * porcentajeVal/100) + indiceTrain
    indiceTest = math.floor(total * porcentajeTest/100) + indiceVal

    Xtrain = X[0 : indiceTrain]
    Ytrain = Y[0 : indiceTrain]

    Xval =  X[indiceTrain : indiceVal]
    Yval = Y[indiceTrain : indiceVal]

    Xtest =  X[indiceVal : indiceTest]
    Ytest = Y[indiceVal : indiceTest]

    return Xtrain, Ytrain, Xval, Yval, Xtest, Ytest

def lecturaDatos(archivo):
    valores = carga_csv(archivo)

    porcentajeTrain = 60
    porcentajeVal = 30
    porcentajeTest = 10
     
    X = valores[:, 0:5]
    Y = valores[:, 5]
    return fraccionar(X, Y, porcentajeTrain, porcentajeVal, porcentajeTest)

def kernelGaussiano(X, y):
    C = 1
    sigma = 0.1
    #HACEMOS RAVEL PORQUE SI NO DA WARNING AL HACER EL FIT
    y = np.ravel(y)
    svm = SVC( kernel='rbf', C=C, gamma = 1/(2 * sigma ** 2))
    svm.fit(X, y)


def eleccionDeParametros(Xtrain, Ytrain, Xval, Yval):
    
    conjunto = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    
    #HACEMOS RAVEL PORQUE SI NO DA WARNING AL HACER EL FIT
    Ytrain = np.ravel(Ytrain)
    Yval = np.ravel(Yval)
    maxAciertos = 0

    for C in conjunto:
        for sigma in conjunto:
            print("C:", C, "Sigma:", sigma)
            svm = SVC( kernel='rbf', C=C, gamma = 1/(2 * sigma ** 2))

            svm.fit(Xtrain, Ytrain)

            Ycal = svm.predict(Xval).reshape(np.shape(Yval))
            aciertosActuales = np.sum(Yval == Ycal)

            if aciertosActuales > maxAciertos:
                maxAciertos = aciertosActuales
                maxComparador = svm
    
    return maxComparador

def calcularAciertos(Xtest, Ytest, svm):
    Ycal = svm.predict(Xtest)
    aciertos = np.sum(Ytest == Ycal)
    
    print(aciertos/len(Ytest))

def main():
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = lecturaDatos("data/random_data_1m.csv")
    svm = eleccionDeParametros(Xtrain, Ytrain, Xval, Yval)
    calcularAciertos(Xtest, Ytest, svm)

    
main()
