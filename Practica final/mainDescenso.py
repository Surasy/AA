from pandas.io.parsers import read_csv
from sklearn import preprocessing as prep
import math 
import numpy as np
import scipy.optimize as opt

def sigmoide(Z):
    return 1 /(1 + np.e**(-Z))

def calcularParteIzq(Y, H):

    return np.dot(np.log(H), Y)

def calcularParteDer(Y, H):
    return np.dot(np.log(1 - H), (1 - Y))


def funcionCoste(Theta, X, Y):
    H = sigmoide(np.dot(X, Theta))
    return -1 / np.shape(X)[0] * (calcularParteIzq(Y, H) + calcularParteDer(Y, H))

def funcionGradiente(Theta, X, Y):
    H = sigmoide(np.dot(X, Theta))
    return 1 / np.shape(X)[0] * np.dot((H - Y), X)


def funcionCosteRegularizado(Theta, X, Y, landa):
    return funcionCoste(Theta, X, Y) + landa/(2*np.shape(X)[0])*(Theta**2).sum()


def funcionGradienteRegularizado(Theta, X, Y, landa):
    return funcionGradiente(Theta, X, Y) + landa/(np.shape(X)[0])*Theta


def problemaRegularizado(Xtrain, Ytrain, landa, pol):
    poly = prep.PolynomialFeatures(pol)
    Xtrain = poly.fit_transform(Xtrain)

    Theta = np.zeros(np.shape(Xtrain)[1])

    print(np.shape(Xtrain))
    print(np.shape(Theta))
    #print(np.shape(funcionGradienteRegularizado(Theta, Xtrain, Ytrain, landa)))

    resultRegularizado = opt.fmin_tnc(func=funcionCosteRegularizado, x0 = Theta, 
    fprime = funcionGradienteRegularizado, args = (Xtrain, Ytrain, landa), messages=0)
    
    return resultRegularizado[0]




def eleccionOptimo(Xval, Yval):
    mejorAcierto = 0

    for pol in [1,2,3,4,5,6]:
        for landa in [0, 0.3, 0.6, 1, 2, 3]:
            print(pol, landa)




def fraccionar(X, Y, porcentajeTrain, porcentajeVal, porcentajeTest):

    #total = len(X)
    total = 10000

    indiceTrain = math.floor(total * porcentajeTrain/100)
    indiceVal = math.floor(total * porcentajeVal/100) + indiceTrain
    indiceTest = math.floor(total * porcentajeTest/100) + indiceVal


    Xtrain = X[0 : indiceTrain]
    Ytrain = Y[0 : indiceTrain]
    Xval =  Y[indiceTrain : indiceVal]
    Yval = Y[indiceTrain : indiceVal]
    Xtest =  X[indiceVal : indiceTest]
    Ytest = X[indiceVal : indiceTest]

    return Xtrain, Ytrain, Xval, Yval, Xtest, Ytest

def carga_csv(file_name):
    # S -> plantarse 0, H -> pedir carta 1
    # W -> ganar 1, L -> perder 0
    valores = read_csv(file_name, header=0).values
    valores[:, 4] = (valores[:, 4] == "H")
    valores[:, 5] = (valores[:, 5] == "W")
    print(valores[0])
    return valores.astype(float)

def lecturaDatos(archivo):
    valores = carga_csv(archivo)

    
    
    porcentajeTrain = 60
    porcentajeVal = 20
    porcentajeTest = 20
    
    
    X = valores[:, 0:5]
    Y = valores[:, 5]
    return fraccionar(X, Y, porcentajeTrain, porcentajeVal, porcentajeTest)

def normalizar(X):
    if np.shape(X)[1] > 1:
        mu = np.mean(X, axis = 0)
        sigma = np.std(X, axis = 0)
        X_norm = (X - mu)/sigma
        return X_norm

    return X


def main():

    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = lecturaDatos("data/random_data_1m.csv")

    Xtrain = normalizar(Xtrain)
    print(problemaRegularizado(Xtrain, Ytrain, 1, 1))
    
    

main()