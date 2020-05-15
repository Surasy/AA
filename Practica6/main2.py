from scipy.io import loadmat
import numpy as np
import scipy.optimize as opt
import codecs as cds
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from data.process_email import email2TokenList
from data.get_vocab_dict import getVocabDict
import math


## easy 2551
## hard 250
## spam 500

## 60% -> 1530,6 easy + 150 hard + 300 spam
## 20% -> 510,2 easy + 50 hard + 100 spam

def cargarDatos(rangoini, rangofin, nombreArchivo, diccionario): 

    Xtrain = np.empty((0, len(diccionario) + 1))

    for i in np.arange(rangoini, rangofin):
        contador = f"{i:04d}"
        easyham = cds.open("data/"+nombreArchivo+"/" + contador +".txt", "r",encoding = "utf-8", errors = "ignore").read()
        easyhamTrain = email2TokenList(easyham)
        processedLine = np.zeros(len(diccionario) + 1)

        #print(easyhamTrain[0])
        for pal in easyhamTrain :
            if pal in diccionario:
                processedLine[diccionario[pal]] = 1
        
        Xtrain = np.vstack((Xtrain, processedLine))
        
    return Xtrain

## easy 2551
## hard 250
## spam 500

def calcularPorcentajes(porcentaje):
    elementosEasy = 2551
    elementosHard = 250
    elementosSpam = 500

    indiceEasy= elementosEasy * porcentaje/100 
    indiceHard =  elementosHard * porcentaje/100 
    indiceSpam = elementosSpam * porcentaje/100
    
    return math.floor(indiceEasy), math.floor(indiceHard), math.floor(indiceSpam)



def iniDatos():
    diccionario = getVocabDict()
    porcentajeTrain = 20
    porcentajeVal = 10
    porcentajeTest = 50


    indiceXEasy, indiceXHard, indiceXSpam = calcularPorcentajes(porcentajeTrain)


    print("Leyendo datos para train...")
    ytrain = np.zeros(indiceXEasy - 1 + indiceXHard - 1 ) 
    Xtrain = cargarDatos(1, indiceXEasy, "easy_ham", diccionario)
    Xtrain = np.vstack((Xtrain, cargarDatos(1, indiceXHard, "hard_ham", diccionario)))
    Xtrain = np.vstack((Xtrain, cargarDatos(1, indiceXSpam, "spam", diccionario)))
    ytrain = np.hstack((ytrain, np.ones(indiceXSpam - 1 )))


    

    print("Leyendo datos para Val...")
    indiceValEasy, indiceValHard, indiceValSpam = calcularPorcentajes(porcentajeTrain + porcentajeVal)
    yval= np.zeros(indiceValEasy - indiceXEasy + indiceValHard - indiceXHard)
    Xval = cargarDatos(indiceXEasy, indiceValEasy, "easy_ham", diccionario)
    Xval = np.vstack((Xval, cargarDatos(indiceXHard, indiceValHard, "hard_ham", diccionario)))
    Xval = np.vstack((Xval, cargarDatos(indiceXSpam, indiceValSpam, "spam", diccionario)))
    yval = np.hstack((yval, np.ones(indiceValSpam - indiceXSpam)))
    print(np.shape(Xval), np.shape(yval))

    print("Leyendo datos para Test...")
    indiceTestEasy, indiceTestHard, indiceTestSpam = calcularPorcentajes(porcentajeTrain + porcentajeVal + porcentajeTest)
    ytest= np.zeros(indiceTestEasy - indiceValEasy + indiceTestHard - indiceValHard)
    Xtest = cargarDatos(indiceValEasy,indiceTestEasy, "easy_ham", diccionario)
    Xtest = np.vstack((Xtest, cargarDatos(indiceValHard, indiceTestHard, "hard_ham", diccionario)))
    Xtest = np.vstack((Xtest, cargarDatos(indiceValSpam, indiceTestSpam, "spam", diccionario)))
    ytest = np.hstack((ytest, np.ones(indiceTestSpam - indiceValSpam)))



    return Xtrain, ytrain, Xval, yval, Xtest, ytest
     

def eleccionDeParametros(X,y, Xval, yval):
    print("Probando y eligiendo parametros...")
    conjunto = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    
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

    return maxComparador

def ratioAciertos(Xtest, ytest, svm):
    print("Calculando ratio de aciertos...")
    ycal = svm.predict(Xtest)
    aciertosActuales = np.sum(ytest == ycal)
    print(aciertosActuales/len(ytest)*100, "%")



def main():
    Xtrain, ytrain, Xval, yval, Xtest, ytest = iniDatos()
    svm = eleccionDeParametros(Xtrain,ytrain, Xval, yval)
    ratioAciertos(Xtest, ytest, svm)


main()