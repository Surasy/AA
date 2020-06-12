from pandas.io.parsers import read_csv
import math
import numpy as np

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



def lecturaDatos(archivo):
    valores = read_csv(archivo, header=1).values
    # S -> plantarse 0, H -> pedir carta 1
    # W -> ganar 1, L -> perder 0

    
    porcentajeTrain = 60
    porcentajeVal = 20
    porcentajeTest = 20
    
    valores[:, 4] = valores[:, 4] == "H"
    valores[:, 5] = valores[:, 5] == "W"
    X = valores[:, 0:5]
    Y = valores[:, 5]
    return fraccionar(X, Y, porcentajeTrain, porcentajeVal, porcentajeTest)





def main():
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = lecturaDatos("data/random_data_1m.csv")
    

main()