# Practica realizada por Daniel Padilla y Sofia Prieto
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pandas.io.parsers import read_csv
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def carga_csv(file_name):
    """carga el fichero csv especificado y lo
    devuelve en un array de numpy
    """
    valores = read_csv(file_name, header=None).values
    # suponemos que siempre trabajaremos con float
    return valores.astype(float)


def dibujarGraficasCoste(X, Y, Thetas):

    mX = np.arange(-10, 10, 0.1)
    mY = np.arange(-1, 4, 0.1)

    mX, mY = np.meshgrid(mX, mY)

    mZ = np.empty_like(mX)
    for ix, iy in np.ndindex(mX.shape):
        mZ[ix, iy] = J([mX[ix, iy], mY[ix, iy]], X, Y)

    """
    Para girar la grafica 3d ax.view_init(elev = 15, azim = 230)
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(elev = 15, azim = 230)


    surf = ax.plot_surface(mX, mY, mZ, cmap = cm.cool, linewidth=0, antialiased=False)
    plt.show()
"""
    plt.figure()
    plt.contour(mX, mY, mZ, np.logspace(-2, 3, 20), cmap = cm.cool)
    plt.scatter(Thetas[0], Thetas[1], marker='x', c= "red")
    plt.show()"""


def descenso_gradiente(X, Y, alpha):
    Thetas = np.array(np.zeros(np.shape(X)[1]))
    Thetas_aux = Thetas

    # Para dibujar graficas en 2D
    """
    plt.figure()
    plt.scatter(X[:, -1], Y, c="#142d4c", marker="x")
    """
    costes = list()
    for i in range(1500):
        H = np.dot(X, Thetas)
        sumatorio = H - Y
        Thetas_aux = Thetas - alpha/len(X)*(np.dot(sumatorio,X))
        Thetas= Thetas_aux
        costes.append((sumatorio**2).sum() /(2*len(X)))

    
    """
    minimo = np.amin(X[:,-1])
    maximo = np.amax(X[:,-1])
    plt.plot([minimo,maximo],[ Thetas[0] + Thetas[1]*minimo,Thetas[0] + Thetas[1]*maximo], c="#9fd3c7")
    plt.show()
    """


    return Thetas, costes



def normalizar(X):
    if np.shape(X)[1] > 1:
        mu = np.mean(X, axis = 0)
        sigma = np.std(X, axis = 0)
        X_norm = (X - mu)/sigma
        return X_norm

    return X


def CalculoThetasVectorizada(X, Y):
    Xtras = np.transpose(X)
    Thetas = np.dot(np.dot(np.linalg.inv(np.dot(Xtras, X)), Xtras),Y)
    return Thetas


def J(Thetas, X, Y):
    H = np.dot(X, Thetas)
    sumatorio = H - Y
    return (sumatorio**2).sum() /(2*len(X))
    

def CalculoCosteVectorizado(Thetas, X, Y):
    N = np.dot(X, Thetas) - Y
    return np.dot(np.transpose(N), N) /(2*len(X))



def main():
    datos = carga_csv("ex1data2.csv")
    X = datos[:, :-1]
    Y = datos[:, -1]

    m = np.shape(X)[0]
    n = np.shape(X)[1]
    

    #Para comprobar sin normalizar
    X_unos = np.hstack([np.ones([m, 1]), X])
    ThetasVectorizadas = CalculoThetasVectorizada(X_unos,Y)
    CosteVectorizado = CalculoCosteVectorizado(ThetasVectorizadas, X_unos, Y)
    solOtra = np.dot(X_unos, ThetasVectorizadas)

    #print ("Coste calculado de forma vectorizada:", CosteVectorizado)
    #print (solOtra)

    #Para usar normalizado
    X = normalizar(X)
    X = np.hstack([np.ones([m, 1]), X])
    alpha = 0.01

    
    #Thetas, costes = descenso_gradiente(X, Y, alpha)
    plt.figure()
    for i in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]:
        Thetas, costes = descenso_gradiente(X, Y, i)
        #print(i, "->", costes[-1])
        plt.plot(np.arange(1500) + 1, costes, label = i)

    plt.legend()
    plt.show()
    solNormalizado  = np.dot(X, Thetas)
    print (solNormalizado - solOtra)


    #dibujarGraficasCoste(X, Y, Thetas)






main()