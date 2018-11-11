# Solution set for CS 155 Set 6, 2016/2017
# Authors: Fabian Boemer, Sid Murching, Suraj Nair

import numpy as np

def grad_U(Ui, Yij, Vj, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    
    return eta * (reg * Ui - Vj * (Yij - Ui.dot(Vj)))


def grad_V(Vj, Yij, Ui, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    return eta * (reg * Vj - Ui * (Yij - Ui.dot(Vj)))

def get_err(U, V, Y, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """
    reg_part = reg / 2 * (np.linalg.norm(U)**2 + np.linalg.norm(V)**2)

    result = 0
    for pt in Y:
        i,j = int(pt[0]) - 1, int(pt[1]) - 1
        result += (pt[2] - U[i].dot(V[j])) ** 2

    return (reg_part +  result/2) / len(Y)

def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300):
    """
    Given a training data matrix Y containing rows (i, j, Y_ij)
    where Y_ij is user i's rating on movie j, learns an
    M x K matrix U and N x K matrix V such that rating Y_ij is approximated
    by (UV^T)_ij.

    Uses a learning rate of <eta> and regularization of <reg>. Stops after
    <max_epochs> epochs, or once the magnitude of the decrease in regularized
    MSE between epochs is smaller than a fraction <eps> of the decrease in
    MSE after the first epoch.

    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE
    of the model.
    """

    

    

    U = np.random.uniform(-.5,.5, (M,K))
    V = np.random.uniform(-.5,.5, (N,K))
    loss_i = get_err(U, V, Y, reg = reg)
    del_t = 1
    del_i = 1
    epoch = 0
    print("initial loss: " + str(loss_i))

    while (del_t / del_i > eps and epoch <= 300):
        print (epoch)
        Y_perm = np.random.permutation(Y)
        old_loss = get_err(U,V,Y,reg=reg)
        for y in Y_perm:
            i, j = int(y[0]) - 1, int(y[1]) - 1
            U[i] -= grad_U(U[i], y[2], V[j], reg, eta)
            V[j] -= grad_V(V[j], y[2], U[i], reg, eta)
    	
        if epoch == 0:
        	del_i = loss_i - get_err(U,V,Y,reg=reg)
        else:
        	del_t = old_loss - get_err(U,V,Y,reg=reg)
        epoch += 1

    return U, V, get_err(U,V,Y,reg = 0)


