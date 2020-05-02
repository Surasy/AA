from scipy.io import loadmat
import numpy as np
import scipy.optimize as opt
from matplotlib import pyplot as plt

#NO USADO?
def sigmoide(Z):
    return 1 /(1 + np.e**(-Z))

def costeSinReguralizar(Thetas, X, y):
    m = len(y)

    H = np.dot(X, np.transpose(Thetas))
    return 1/(2*m)*np.sum((H - y)**2)


def costeRegularizado(Thetas, X, y, landa):
    m = len(y)
    parteIzq = costeSinReguralizar(Thetas, X, y)
    parteDer = landa/(2*m)*np.sum(Thetas[:, 1:]**2)
    return  parteIzq + parteDer
    

    
def gradienteRegularizada(Thetas, X, y, landa):
    m = len(y)
    H = np.dot(X, np.transpose(Thetas))

    gradiente =  1/m*np.dot((H - y).T, X)

    gradiente[:, 1:] = gradiente[:,1:] + landa/ np.shape(X)[0] * Thetas[:, 1:]
   

    return gradiente

def calculoCosteYGradiente(Thetas, X, y, landa):
    Thetas = np.reshape(Thetas, (1,np.shape(X)[1]))
    return costeRegularizado(Thetas, X, y, landa), gradienteRegularizada(Thetas, X, y, landa) 

def inicializarDatos(archivo):
    valores = loadmat(archivo)
    X, y = valores['X'], valores['y']
    Xval, yval = valores['Xval'], valores['yval']
    Xtest, ytest = valores['Xtest'], valores['ytest']

    return X, y, Xval, yval, Xtest, ytest

def iniThetas(shape):
    Thetas = np.ones((1,shape[1]))
    return Thetas

def dibujarRegresionLineal(X, y, Thetas):
    plt.figure()
    plt.scatter(X[:, 1:], y, marker= 'x')

    indiceMinimo = np.argmin(X[:,1])
    indiceMaximo = np.argmax(X[:,1])
    
    plt.plot([X[indiceMinimo][1], X[indiceMaximo][1]], 
        [np.dot(X[indiceMinimo], np.transpose(Thetas)), np.dot(X[indiceMaximo], np.transpose(Thetas))])

    plt.show()

def optimiza(Thetas, X, y, landa):
    Thetas = Thetas.ravel()
    returned = opt.minimize(fun = calculoCosteYGradiente, x0 = Thetas, args = (X, y, landa), 
        method='TNC', jac=True, options={'maxiter': 70})
    return returned["x"]

def curvasDeAprendizaje(X, y, Xval, yval, Thetas, landa):
    
    errorTest = []
    errorValidation = []
    for i in (np.arange(np.shape(X)[0])+ 1):
        
        ThetasOpt = optimiza(Thetas, X[0 : i], y[0 : i], landa)

        errorTest.append(costeSinReguralizar(ThetasOpt, X[0 : i], y[0 : i]))
        errorValidation.append(costeSinReguralizar(ThetasOpt, Xval, yval))

    
    dibujarCurvasDeAprendizaje(errorTest, errorValidation)

    

def dibujarCurvasDeAprendizaje(errorTest, errorValidation): 
    print()
    plt.figure()

    plt.plot(np.arange(12) + 1, errorTest)
    plt.plot(np.arange(12) + 1, errorValidation)
    plt.show()

def regresionLineal(Thetas, X, y, landa):

    ThetasOpt = optimiza(Thetas, X, y, landa)    
    dibujarRegresionLineal(X, y, ThetasOpt)
 
def main():
    X, y, Xval, yval, Xtest, ytest = inicializarDatos('data\ex5data1.mat')

    #Xval hstack
    Xval = np.hstack([np.ones([np.shape(Xval)[0], 1]), Xval])
    X = np.hstack([np.ones([np.shape(X)[0], 1]), X])

    Thetas = iniThetas(np.shape(X))
    landa = 1



    #regresionLineal(Thetas, X, y, landa)
    curvasDeAprendizaje(X, y, Xval, yval, Thetas, landa)


    


main()