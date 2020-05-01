from scipy.io import loadmat
import numpy as np
import scipy.optimize as opt

#NO USADO?
def sigmoide(Z):
    return 1 /(1 + np.e**(-Z))

def costeRegularizado(X, y, Thetas, landa):
    m = len(y)

    H = np.dot(X, np.transpose(Thetas))
    parteIzq = 1/(2*m)*np.sum((H - y)**2)
    parteDer = landa/(2*m)*np.sum(Thetas[:, 1:]**2)
    return  parteIzq + parteDer
    
    
def gradienteRegularizada(X, y, Thetas, landa):
    m = len(y)
    H = np.dot(X, np.transpose(Thetas))

    #gradiente =  1/m*np.sum(H - y)

    #gradiente =  1/m*np.dot(np.sum(H - y), X)
    gradiente =  1/m*np.dot((H - y).T, X)

    gradiente[:, 1:] = gradiente[:,1:] + landa/ np.shape(X)[0] * Thetas[:, 1:]
   

    return gradiente


def inicializarDatos(archivo):
    #(21, 1) (21, 1) (21, 1) (21, 1)
    valores = loadmat(archivo)
    X, y = valores['X'], valores['y']
    Xval, yval = valores['Xval'], valores['yval']
    Xtest, ytest = valores['Xtest'], valores['ytest']



    return X, y, Xval, yval, Xtest, ytest

def iniThetas(shape):
    Thetas = np.ones((1,shape[1]))
    return Thetas

def main():
    X, y, Xval, yval, Xtest, ytest = inicializarDatos('data\ex5data1.mat')

    X = np.hstack([np.ones([np.shape(X)[0], 1]), X])

    Thetas = iniThetas(np.shape(X))

    landa = 1

    
    #Xtest = np.hstack([np.ones([np.shape(Xtest)[0], 1]), Xtest])
    print(costeRegularizado(X, y, Thetas, landa))
    print(gradienteRegularizada(X, y, Thetas, landa))
    #print(costeRegularizado(Xtest, ytest, Thetas, landa))
    


main()