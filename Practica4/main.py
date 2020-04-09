from scipy.io import loadmat
import numpy as np
import 


def sigmoideDevivada(Z):
    return sigmoide(Z) * (1 - sigmoide(Z))

def sigmoide(Z):
    return 1 /(1 + np.e**(-Z))

def calcularParteIzq(Y, H):
    return -1 * np.log(H) * Y

def calcularParteDer(Y, H):
    return -1 * np.log(1 - H) * (1 - Y)


def funcionCoste(capaFinal, tamX, Y):
    
    return np.sum(1 / tamX * (calcularParteIzq(Y, capaFinal) + calcularParteDer(Y, capaFinal)))

def funcionCosteRegularizada(theta1, theta2,capaFinal, tamX, Y, landa):
    return funcionCoste(capaFinal, tamX, Y) + ( landa/(2 * tamX) * (np.sum(theta1 ** 2) + np.sum(theta2 ** 2)))

def  pesosAleatorios(L_ini, L_out):
    Eini = 0.12

    pesos = np.random.randint(low=-Eini *100, high=Eini*100 ,size=(L_out, L_ini + 1))

    return pesos/100





def backprop (params_rn , num_entradas , num_ocultas , num_etiquetas , X, y , reg):
    theta1 = np.reshape(params_rn[:num_ocultas, (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
    theta2 = np.reshape(params_rn[num_ocultas, (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))
   
    X = np.hstack([np.ones([np.shape(X)[0], 1]), X])
    gradiente = 0

    z2 = sigmoide(np.dot(X, np.transpose(theta1)))
    a2 = np.hstack([np.ones([np.shape(z2)[0], 1]), z2])
    a3 = sigmoide(np.dot(a2, np.transpose(theta2)))
    
   
    for i in range (np.shape(X)[0]):
        deltaTres = a3[i] - y[i]
        deltaDos = np.dot(np.transpose(theta2), deltaTres) * sigmoideDevivada(z2[i])
        gradientePrimero = gradientePrimero + np.dot(deltaDos[1:],  X[i])
        gradienteSegundo = gradienteSegundo + np.dot(deltaTres,  a2[i])
   
   
    gradiente = (1/ np.shape(X)[0]) * (gradientePrimero+gradienteSegundo)


    #return coste, gradiente 



def cargarDatos(nombre):
    weights = loadmat(nombre)
    theta1, theta2 = weights['Theta1'], weights['Theta2']
    # Theta1 25 x 401
    # Theta2 10 x 26
    return theta1, theta2


def main():
    data = loadmat('data\ex4data1.mat')
    y = data['y'].ravel() # (5000, 1) --> (5000,)
    X = data['X']
    m = len(y)
    input_size = X.shape[1]
    num_labels = 10
    y = (y - 1)
    y_onehot = np.zeros((m, num_labels))  # 5000 x 10
    for i in range(m):
        y_onehot[i][y[i]] = 1

    landa = 1

    theta1, theta2 = cargarDatos('data\ex4weights.mat')

    print(pesosAleatorios(2,4))
    #print(funcionCosteRegularizada(theta1, theta2, X, y_onehot, landa))
    
    


def debugInitializeWeights(fan_in, fan_out):
    """
    Initializes the weights of a layer with fan_in incoming connections and
    fan_out outgoing connections using a fixed set of values.
    """

    # Set W to zero matrix
    W = np.zeros((fan_out, fan_in + 1))

    # Initialize W using "sin". This ensures that W is always of the same
    # values and will be useful in debugging.
    W = np.array([np.sin(w) for w in
                  range(np.size(W))]).reshape((np.size(W, 0), np.size(W, 1)))

    return W


def computeNumericalGradient(J, theta):
    """
    Computes the gradient of J around theta using finite differences and
    yields a numerical estimate of the gradient.
    """

    numgrad = np.zeros_like(theta)
    perturb = np.zeros_like(theta)
    tol = 1e-4

    for p in range(len(theta)):
        # Set perturbation vector
        perturb[p] = tol
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)

        # Compute numerical gradient
        numgrad[p] = (loss2 - loss1) / (2 * tol)
        perturb[p] = 0

    return numgrad


def checkNNGradients(costNN, reg_param):
    """
    Creates a small neural network to check the back propogation gradients.
    Outputs the analytical gradients produced by the back prop code and the
    numerical gradients computed using the computeNumericalGradient function.
    These should result in very similar values.
    """
    # Set up small NN
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # Generate some random test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)

    # Reusing debugInitializeWeights to get random X
    X = debugInitializeWeights(input_layer_size - 1, m)

    # Set each element of y to be in [0,num_labels]
    y = [(i % num_labels) for i in range(m)]

    ys = np.zeros((m, num_labels))
    for i in range(m):
        ys[i, y[i]] = 1

    # Unroll parameters
    nn_params = np.append(Theta1, Theta2).reshape(-1)

    # Compute Cost
    cost, grad = costNN(nn_params,
                        input_layer_size,
                        hidden_layer_size,
                        num_labels,
                        X, ys, reg_param)

    def reduced_cost_func(p):
        """ Cheaply decorated nnCostFunction """
        return costNN(p, input_layer_size, hidden_layer_size, num_labels,
                      X, ys, reg_param)[0]

    numgrad = computeNumericalGradient(reduced_cost_func, nn_params)

    # Check two gradients
    np.testing.assert_almost_equal(grad, numgrad)
    return (grad - numgrad)




main()