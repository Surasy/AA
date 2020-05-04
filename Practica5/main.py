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
        method='L-BFGS-B', jac=True)
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

def regresionLinealNormalizada(X, y, landa):
    p = 8
    X_norm, mu, sigma = polinomizar(X, p)
    X_norm = np.hstack([np.ones([np.shape(X_norm)[0], 1]), X_norm])


    Thetas = iniThetas(np.shape(X_norm))
    ThetasOpt = optimiza(Thetas, X_norm, y, landa)   
    dibujarRegresionLinealNormalizada(X, y, ThetasOpt, mu, sigma, p)

def dibujarRegresionLineal(X, y, Thetas):
    plt.figure()
    plt.scatter(X[:, 1], y, marker= 'x')


    
    indiceMinimo = np.argmin(X[:,1])
    indiceMaximo = np.argmax(X[:,1])

    plt.plot([X[indiceMinimo][1], X[indiceMaximo][1]], 
        [np.dot(X[indiceMinimo], np.transpose(Thetas)), np.dot(X[indiceMaximo], np.transpose(Thetas))])

    plt.show()

def dibujarRegresionLinealNormalizada(X, y, Thetas, mu, sigma, p):
    plt.figure()

    plt.scatter(X[:, 0], y, marker= 'x')
    holgura = 5

    minimo = np.amin(X[:, 0])
    maximo = np.amax(X[:, 0])
    maximo += holgura
    minimo -= holgura
    rango = np.arange(minimo, maximo, 0.05)
    rango_uni = np.zeros((len(rango), 1))
    
    rango_uni[:, 0] = rango[:] 
    #print(rango_uni)

    rango_norm = rango_uni
    for i in np.arange(2, p + 1):
        rango_norm = np.hstack((rango_norm, rango_uni**i))


    rango_norm = (rango_norm - mu)/sigma
    rango_norm = np.hstack([np.ones([np.shape(rango_norm)[0], 1]), rango_norm])
    plt.plot(rango, np.dot(rango_norm, np.transpose(Thetas)))
    print(np.shape(rango_norm), np.shape(Thetas))

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
    Thetas = np.zeros((1,shape[1])) #(1,2)
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


def curvasDeAprendizajeNormalizada(X, y, Xval, yval, landa):
    p = 8

    X_norm, mu, sigma = polinomizar(X, p)
    X_norm = np.hstack([np.ones([np.shape(X_norm)[0], 1]), X_norm])



    Xval_norm = Xval
    for i in np.arange(2, p + 1):
        Xval_norm = np.hstack((Xval_norm, Xval**i))

    Xval_norm = (Xval_norm - mu)/sigma
    Xval_norm = np.hstack([np.ones([np.shape(Xval_norm)[0], 1]), Xval_norm])
    
    print(Xval_norm)
    #print(np.shape(Xval_norm))
    Thetas = iniThetas(np.shape(X_norm))
    curvasDeAprendizaje(X_norm, y, Xval_norm, yval, Thetas, landa)


    
def seleccionDeParametro(X, y, Xval, yval):
    p = 8

    errorTest = []
    errorValidation = []

    X_norm, mu, sigma = polinomizar(X, p)
    X_norm = np.hstack([np.ones([np.shape(X_norm)[0], 1]), X_norm])



    Xval_norm = Xval
    for i in np.arange(2, p + 1):
        Xval_norm = np.hstack((Xval_norm, Xval**i))

    Xval_norm = (Xval_norm - mu)/sigma
    Xval_norm = np.hstack([np.ones([np.shape(Xval_norm)[0], 1]), Xval_norm])


    Thetas = iniThetas(np.shape(X_norm))
    for i in (0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10):
        ThetasOpt = optimiza(Thetas, X_norm, y, i)
        errorTest.append(costeSinReguralizar(ThetasOpt, X_norm, y))
        errorValidation.append(costeSinReguralizar(ThetasOpt, Xval_norm, yval))

    dibujarCurvasDeAprendizaje(errorTest, errorValidation)



def main():
    X, y, Xval, yval, Xtest, ytest = inicializarDatos('data\ex5data1.mat')


    #SOLO SIRVE PARA SIN NORMALIZAR Thetas = iniThetas(np.shape(X_norm))
    landa = 100


    #regresionLinealNormalizada(X, y, landa)
    #curvasDeAprendizajeNormalizada(X, y, Xval, yval, landa)
    seleccionDeParametro(X, y, Xval, yval)
    """
    #Apartado 1 y 2
    Xval = np.hstack([np.ones([np.shape(Xval)[0], 1]), Xval])
    X = np.hstack([np.ones([np.shape(X)[0], 1]), X])

    Thetas = iniThetas(np.shape(X))
    landa = 0


    
    regresionLineal(Thetas, X, y, landa)
    #curvasDeAprendizaje(X, y, Xval, yval, Thetas, landa)
    """





main()