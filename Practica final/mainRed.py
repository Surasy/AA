from pandas.io.parsers import read_csv
import scipy.optimize as opt
import pandas as pd
import numpy as np
import math 

def fraccionar(X, Y, porcentajeTrain, porcentajeVal, porcentajeTest):

    #total = len(X)
    total = 5000

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

def sigmoide(Z):
    return 1 /(1 + np.e**(-Z))

def calcularParteIzq(Y, H):
    return -1 * np.log(H) * Y

def calcularParteDer(Y, H):
    return -1 * np.log(1 - H) * (1 - Y)

def funcionCoste(capaFinal, tamX, Y):
    return np.sum(1 / tamX * (calcularParteIzq(Y, capaFinal) + calcularParteDer(Y, capaFinal)))

def funcionCosteRegularizada(theta1, theta2, capaFinal, tamX, Y, landa):
    return funcionCoste(capaFinal, tamX, Y) + landa/(2 * tamX) * (np.sum(theta1[:, 1:] ** 2) + np.sum(theta2[:, 1:] ** 2))

def carga_csv(file_name):
    # S -> plantarse 0, H -> pedir carta 1
    # W -> ganar 1, L -> perder 0
    valores = read_csv(file_name, header=0).values
    valores[:, 4] = (valores[:, 4] == "H")
    valores[:, 5] = (valores[:, 5] == "W")

    return valores.astype(float)

def backprop (params_rn , num_entradas, num_ocultas, num_etiquetas , X, y , reg):
    theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
    theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))

    gradientePrimero = np.zeros(np.shape(theta1))
    gradienteSegundo = np.zeros(np.shape(theta2))

    X = np.hstack([np.ones([np.shape(X)[0], 1]), X])

    z2 = sigmoide(np.dot(X, np.transpose(theta1)))
    a2 = np.hstack([np.ones([np.shape(z2)[0], 1]), z2])
    a3 = sigmoide(np.dot(a2, np.transpose(theta2)))

    coste = funcionCosteRegularizada(theta1, theta2, a3, np.shape(X)[0], y, reg)
    for i in range (np.shape(X)[0]):
        deltaTres = a3[i] - y[i]
        deltaDos = np.dot(np.transpose(theta2), deltaTres) * a2[i] * (1 - a2[i])
        gradientePrimero = gradientePrimero + np.dot(deltaDos[1:, np.newaxis],  X[i][np.newaxis, :])
        gradienteSegundo = gradienteSegundo + np.dot(deltaTres[:, np.newaxis],  a2[i][np.newaxis, :])

    gradienteUno = 1/ np.shape(X)[0] * gradientePrimero
    gradienteDos = 1/ np.shape(X)[0] * gradienteSegundo  

    gradienteUno[:, 1:] = gradienteUno[:,1:] + reg/ np.shape(X)[0] * theta1[:, 1:]
    gradienteDos[:, 1:] = gradienteDos[:,1:] + reg/ np.shape(X)[0] * theta2[:, 1:]
    
    gradiente = np.concatenate((np.ravel(gradienteUno),np.ravel(gradienteDos)))

    return coste, gradiente 


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
        return X_norm, mu, sigma

    return X


def normalizarDadosDatos(X, mu, sigma):
    if np.shape(X)[1] > 1:
        X_norm = (X - mu)/sigma
        return X_norm

    return X

def pesosAleatorios(L_ini, L_out):
    Eini = 0.12

    pesos = np.random.random((L_out, L_ini + 1))*(2*Eini)-Eini
    return pesos

def comprobar(params_rn , num_entradas, num_ocultas, num_etiquetas , X, y):
    theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
    theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))

    X = np.hstack([np.ones([np.shape(X)[0], 1]), X])

    z2 = sigmoide(np.dot(X, np.transpose(theta1)))
    a2 = np.hstack([np.ones([np.shape(z2)[0], 1]), z2])
    a3 = sigmoide(np.dot(a2, np.transpose(theta2)))


    sol = np.sum(np.argmax(y, axis= 1) == np.argmax(a3, axis= 1))
 
    return sol / np.shape(X)[0]

def oneShootY(Y):
    # W -> 1, 0
    # L -> 0, 1
    oneY = np.zeros((len(Y), 2))
    posicionesWin = np.where(Y == 1)

    posicionesLose = np.where(Y == 0)

    oneY[posicionesWin[0], 0] = 1
    oneY[posicionesLose[0], 1] = 1

    return oneY

def seleccionMejorLanda(params_rn, nodosEntrada, nodosOcultos, nodosSalida, Xtrain, Ytrain, Xval, Yval):
    mejorPorcentaje = 0
    landas = [0, 1, 3, 6, 10]
    
    for landa in landas:
        print("Trabajando con landa:", landa)
        returned = opt.minimize(fun = backprop, x0 = params_rn, 
            args = (nodosEntrada, nodosOcultos, nodosSalida, Xtrain, Ytrain, landa), method = 'TNC', jac = True, options = {'maxiter': 70})
        
        #PARA ESCOGER LAS MEJOR LANDA (NOS QUEDAMOS CON ESAS THETAS)
        sol = comprobar(returned["x"], nodosEntrada, nodosOcultos, nodosSalida , Xval, Yval)
        if sol > mejorPorcentaje:
            mejorPorcentaje = sol
            thetasOpt = returned["x"]
            mejorLanda = landa
        
    #print(mejorPorcentaje, "con landa", mejorLanda)

    #
    theta1 = np.reshape(thetasOpt[:nodosOcultos * (nodosEntrada + 1)], (nodosOcultos, (nodosEntrada + 1)))
    theta2 = np.reshape(thetasOpt[nodosOcultos * (nodosEntrada + 1):], (nodosSalida, (nodosOcultos + 1)))
    print(theta1, theta2)
    #

    datos = pd.DataFrame(data=thetasOpt)
    datos.to_csv("data/mejorRed.csv", index=False)
    return thetasOpt


def cargarThetas(file_name, num_entradas, num_ocultas, num_etiquetas):
    valores = read_csv(file_name, header=0).values
    valores = np.ravel(valores)
    theta1 = np.reshape(valores[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
    theta2 = np.reshape(valores[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))
    return theta1, theta2

def main():
    nodosEntrada = 5
    nodosSalida = 2
    nodosOcultos = 7

    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = lecturaDatos("data/random_data_1m.csv")

    Ytrain = oneShootY(Ytrain)
    Yval = oneShootY(Yval)
    Ytest = oneShootY(Ytest)

    Xtrain, mu, sigma = normalizar(Xtrain)
    Xval = normalizarDadosDatos(Xval, mu, sigma)
    Xtest = normalizarDadosDatos(Xtest, mu, sigma)

    theta1, theta2 = pesosAleatorios(nodosEntrada, nodosOcultos), pesosAleatorios(nodosOcultos, nodosSalida)

    params_rn = np.concatenate((np.ravel(theta1),np.ravel(theta2)))


    thetasOpt = seleccionMejorLanda(params_rn, nodosEntrada, nodosOcultos, nodosSalida, Xtrain, Ytrain, Xval, Yval)
    porcentajeAciertos = comprobar(thetasOpt, nodosEntrada, nodosOcultos, nodosSalida , Xtest, Ytest)
    print("Aciertos:", porcentajeAciertos * 100, "%")
    #
    Thetas1, Thetas2 = cargarThetas("data/mejorRed.csv", nodosEntrada, nodosOcultos, nodosSalida)
    print(Thetas1, Thetas2)
    #


main()