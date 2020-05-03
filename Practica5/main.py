from scipy.io import loadmat
import numpy as np
import scipy.optimize as opt
from matplotlib import pyplot as plt

#NO USADO?
def sigmoide(Z):
    return 1 /(1 + np.e**(-Z))



def costeSinReguralizar(Thetas, X, y):
    m = len(y)
    # ESTAMOS FORZANDO A QUE LAS THETAS QUE ENTREN SEAN (1, 2) DEBIDO A QUE 
    # MINIMIZE FUERZA LAS THETAS A SER (1, 2) Y EN NUESTRO CODIGO USABAMOS LAS 
    # THETAS COMO (2,)

    Thetas = Thetas.reshape(1,np.shape(X)[1])

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

    gradiente[:, 1:] = gradiente[:,1:] + landa/ np.shape(X)[0] * Thetas[:,1:]
   

    return gradiente

def calculoCosteYGradiente(Thetas, X, y, landa):

    Thetas = np.reshape(Thetas, (1,np.shape(X)[1]))
    return costeRegularizado(Thetas, X, y, landa), gradienteRegularizada(Thetas, X, y, landa) 

def optimiza(Thetas, X, y, landa):
    returned = opt.minimize(fun = calculoCosteYGradiente, x0 = Thetas, args = (X, y, landa), 
        method='TNC', jac=True, options={'maxiter': 70})
    return returned["x"]



def curvasDeAprendizaje(X, y, Xval, yval, Thetas, landa):
    
    errorTest = []
    errorValidation = []



    for i in (np.arange(1, len(X) + 1)):
        ThetasOpt = optimiza(Thetas, X[0 : i], y[0 : i], landa)
        errorTest.append(costeSinReguralizar(ThetasOpt, X[0 : i], y[0 : i]))
        errorValidation.append(costeSinReguralizar(ThetasOpt, Xval, yval))


    dibujarCurvasDeAprendizaje(errorTest, errorValidation)

def regresionLineal(Thetas, X, y, landa):
    ThetasOpt = optimiza(Thetas, X, y, landa)   
    dibujarRegresionLineal(X, y, ThetasOpt)

def regresionLinealNormalizada(Thetas, X, y, landa, mu, sigma):
    ThetasOpt = optimiza(Thetas, X, y, landa)   
    dibujarRegresionLinealNormalizada(X, y, ThetasOpt, mu, sigma)

def dibujarRegresionLineal(X, y, Thetas):
    plt.figure()
    plt.scatter(X[:, 1], y, marker= 'x')


    
    indiceMinimo = np.argmin(X[:,1])
    indiceMaximo = np.argmax(X[:,1])

    plt.plot([X[indiceMinimo][1], X[indiceMaximo][1]], 
        [np.dot(X[indiceMinimo], np.transpose(Thetas)), np.dot(X[indiceMaximo], np.transpose(Thetas))])

    plt.show()

def dibujarRegresionLinealNormalizada(X, y, Thetas, mu, sigma):
    plt.figure()
    plt.scatter(X[:, 1], y, marker= 'x')


    minimo = np.amin(X[:, 1])
    maximo = np.amax(X[:, 1])
    rango = np.arange(minimo, maximo, 0.05)
    #¿?¿?¿
    plt.plot(rango, )
    
    plt.show()


def dibujarCurvasDeAprendizaje(errorTest, errorValidation): 

    plt.figure()

    plt.plot(np.arange(1, len(errorTest) + 1), errorTest)
    plt.plot(np.arange(1, len(errorValidation) + 1), errorValidation)
    plt.show()



def inicializarDatos(archivo):
    valores = loadmat(archivo)
    X, y = valores['X'], valores['y']
    Xval, yval = valores['Xval'], valores['yval']
    Xtest, ytest = valores['Xtest'], valores['ytest']

    return X, y, Xval, yval, Xtest, ytest

def iniThetas(shape):
    Thetas = np.ones((1,shape[1])) #(1,2)
    #Thetas = np.ones(shape[1]) #(2,)

    return Thetas
 

def normalizar(X):
    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X_norm = (X - mu)/sigma


    
    return X_norm, mu, sigma
    

def polinomizar(X, p):
    polin = X

    for i in np.arange(2, p + 1):
        polin = np.hstack((polin, X**i))

    return normalizar(polin)



def main():
    X, y, Xval, yval, Xtest, ytest = inicializarDatos('data\ex5data1.mat')

    X_norm, mu, sigma = polinomizar(X, 2)
    X_norm = np.hstack([np.ones([np.shape(X_norm)[0], 1]), X_norm])


    Thetas = iniThetas(np.shape(X_norm))
    landa = 0


    regresionLinealNormalizada(Thetas, X_norm, y, landa, mu, sigma)
    """
    #Apartado 1 y 2
    Xval = np.hstack([np.ones([np.shape(Xval)[0], 1]), Xval])
    X = np.hstack([np.ones([np.shape(X)[0], 1]), X])

    Thetas = iniThetas(np.shape(X))
    landa = 0


    
    regresionLineal(Thetas, X, y, landa)
    #curvasDeAprendizaje(X, y, Xval, yval, Thetas, landa)
    """


"""
opt.minimize utiliza para las operaciones thetas con shape (1,2) pero la theta
optimizada devuelve un shape (2,). ¿?¿?¿?¿?¿?

"""


main()