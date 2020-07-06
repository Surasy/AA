from pandas.io.parsers import read_csv
import pandas as pd
from sklearn import preprocessing as prep
import math 
import numpy as np
import scipy.optimize as opt
from matplotlib import pyplot as plt

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

def problemaRegularizado(Xtrain, Ytrain, landa):
    Theta = np.zeros(np.shape(Xtrain)[1])

    resultRegularizado = opt.fmin_tnc(func=funcionCosteRegularizado, x0 = Theta, 
    fprime = funcionGradienteRegularizado, args = (Xtrain, Ytrain, landa), messages=0)
    
    return resultRegularizado[0]


def numeroAciertos(ThetasOpt, Xval, Yval):
    H = sigmoide(np.dot(Xval, ThetasOpt))
    H = H >= 0.5
    return np.sum(H == Yval)

def dibujarSeleccionDeParametro(errorValidation, landas, polinomio):
    plt.plot(landas, errorValidation, label= "Pol: " + str(polinomio))
    plt.legend()
    


def eleccionOptimo(Xtrainlimpio, Ytrain, Xvallimpio, Yval, Xtest, Ytest):
    mejorAcierto = 0
    mejorPol = 0

    landas = [0, 1, 3, 6, 10]
    plt.figure()
    for pol in [1,2,3,4,5]:
        errorValidation = []
        
        for landa in landas:
            print("Landa:", landa, " Pol:", pol)
            poly = prep.PolynomialFeatures(pol)
            Xval = poly.fit_transform(Xvallimpio)
            Xtrain = poly.fit_transform(Xtrainlimpio)
        
            ThetasOpt = problemaRegularizado(Xtrain, Ytrain, landa)
            aciertosActuales = numeroAciertos(ThetasOpt, Xval, Yval)

            errorValidation.append(aciertosActuales/ len(Xval)*100)

            if  aciertosActuales > mejorAcierto:
                mejorLanda = landa
                mejorPol = pol
                ThetasMejor = ThetasOpt
                mejorAcierto = aciertosActuales

        dibujarSeleccionDeParametro(errorValidation, landas, pol)

    plt.title("% aciertos para diferentes lamdas según su polinomio")
    plt.xlabel("lamda")
    plt.ylabel("% aciertos")
    plt.show()

    poly = prep.PolynomialFeatures(mejorPol)
    Xtest = poly.fit_transform(Xtest)
    aciertos = numeroAciertos(ThetasMejor, Xtest, Ytest)
    print("Mejor Landa:", mejorLanda, "Mejor Pol", mejorPol)
    print(aciertos)
    print("Porcentaje de aciertos: " , aciertos/len(Xval)*100)

    print(ThetasMejor)

    #DESCOMENTAR PARA GUARDAR LAS TETHAS EN UN FICHERO CSV
    guardarDatos(ThetasMejor, mejorPol)


    


def fraccionar(X, Y, porcentajeTrain, porcentajeVal, porcentajeTest):
    total = len(X)
    #total = 10000

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

def carga_csv(file_name):
    # S -> plantarse 0, H -> pedir carta 1
    # W -> ganar 1, L -> perder 0
    valores = read_csv(file_name, header=0).values
    valores[:, 4] = (valores[:, 4] == "H")
    valores[:, 5] = (valores[:, 5] == "W")

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
        return X_norm, mu, sigma

    return X


def normalizarDadosDatos(X, mu, sigma):
    if np.shape(X)[1] > 1:
        X_norm = (X - mu)/sigma
        return X_norm

    return X


#FUNCION PARA PODER CARGAR UNAS THETAS QUE TENGAMOS EN UN FICHERO CSV
def cargarMejoresDatos(ficheroThetas, ficheroPol):
    valores = read_csv(ficheroThetas, header=0).values
    valores = np.ravel(valores)
    file1 = open(ficheroPol,"r") 

    return valores.astype(float), int(file1.read())

#FUNCION PARA GUARDAR LAS THETAS Y POLINOMIO QUE MEJOR RESULTADOS HAN DADO
def guardarDatos(ThetasMejor, mejorPol):
    datos = pd.DataFrame(data=ThetasMejor)
    datos.to_csv("data/mejorDescenso.csv", index=False)

    file1 = open("data/mejorPolDescenso.txt","w") 
    file1.write(str(mejorPol))
    file1.close() 

    
def main():
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = lecturaDatos("data/random_data_1m.csv")

    Xtrain, mu, sigma = normalizar(Xtrain)
    Xval = normalizarDadosDatos(Xval, mu, sigma)
    Xtest = normalizarDadosDatos(Xtest, mu, sigma)
   
    eleccionOptimo(Xtrain, Ytrain, Xval, Yval, Xtest, Ytest)
    
    
    #Thetas, Pol = cargarMejoresDatos("data/mejorDescenso.csv", "data/mejorPolDescenso.txt")

    
    
    
    

main()