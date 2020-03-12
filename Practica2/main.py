import numpy as np
from matplotlib import pyplot as plt
from pandas.io.parsers import read_csv
import scipy.optimize as opt
from sklearn import preprocessing as prep

def carga_csv(file_name):
    """carga el fichero csv especificado y lo
 devuelve en un array de numpy
    """
    valores = read_csv(file_name, header=None).values
    # suponemos que siempre trabajaremos con float
    return valores.astype(float)

def visualizacionDatos(X, Y, Thetas, regularizada, poly):
    plt.figure()
    posx = np.where(Y == 1)
    posp = np.where(Y == 0)
    plt.scatter(X[posx, 0], X[posx, 1], marker='*', c='#71c7ec')
    plt.scatter(X[posp, 0], X[posp, 1], c='#005073')


    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

    if regularizada:
        h = sigmoide(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(Thetas))
    else:
        h = sigmoide(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(Thetas))

    h = h.reshape(xx1.shape)

    # el cuarto parámetro es el valor de z cuya frontera se
    # quiere pintar
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='black')
    plt.show()
    plt.close()

def calculoPorcentajeCorrectas(X, Y, Thetas):
    H = sigmoide(np.dot(X, Thetas))
    #IGUAL ES UN TRUE?
    binarios = (H >= 0.5)
    listaaciertos = (binarios == Y)

    return listaaciertos.sum()/len(listaaciertos)


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


#  ********************* PARTE 2 ************************

def funcionCosteRegularizado(Theta, X, Y, landa):
    return funcionCoste(Theta, X, Y) + landa/(2*np.shape(X)[0])*(Theta**2).sum()


def funcionGradienteRegularizado(Theta, X, Y, landa):
    return funcionGradiente(Theta, X, Y) + landa/(np.shape(X)[0])*Theta


def problemaSinRegularizar(X, Y):

    X = np.hstack([np.ones([np.shape(X)[0], 1]), X])
    #Parte 1
    Theta = np.zeros(np.shape(X)[1])
    result = opt.fmin_tnc(func = funcionCoste, x0 = Theta, fprime = funcionGradiente, args = (X, Y), messages=0)
    theta_opt = result[0]

    print('Porcentaje de aciertos: ', calculoPorcentajeCorrectas(X, Y, theta_opt))
    
    print('Coste de la función: ', funcionCoste(theta_opt, X, Y)) #comprobacion de coste correcto

    visualizacionDatos(X[:, 1:], Y, theta_opt, False, None)


def problemaRegularizado (X,Y):

    landa = 1

    poly = prep.PolynomialFeatures(6)
    XReg = poly.fit_transform(X)
    Theta = np.zeros(np.shape(XReg)[1])
  
    print(funcionCosteRegularizado(np.zeros(np.shape(XReg)[1]), XReg, Y, 1))
    resultRegularizado = opt.fmin_tnc(func = funcionCosteRegularizado, x0 = Theta, fprime = funcionGradienteRegularizado, args = (XReg, Y, landa), messages=0)
    thetas_opt_reg = resultRegularizado[0]
    #XREG?¿
    visualizacionDatos(X, Y, thetas_opt_reg, True, poly)



def main():
    datos = carga_csv('ex2data2.csv')
    X = datos[:, :-1]
    np.shape(X)       
    Y = datos[:, -1]
    np.shape(Y)         
    m = np.shape(X)[0]
    n = np.shape(X)[1] 


    #problemaSinRegularizar(X,Y)

    problemaRegularizado(X,Y)




main()
