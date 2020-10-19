import numpy as np
from numba import cuda
from numpy.linalg import lstsq

@cuda.jit
def multiple_regression(A,X,Y):
	i,j = cuda.grid(2)
	if i < X.shape[1].size:
		if j < X.shape[2].size:
			x = X[:,i,j,:]   # variable, x, y, time
			y = Y[i,j,:]     # x, y, time
			A[:,i,j] = lstsq(x,y)


def fast_regression(X,y):

    X = X.values
    newX = np.ones([X.shape[0]+1,X.shape[1]])
    newX[:-1,:] = X
    X = newX.transpose()
    y = y.values

    p = np.empty([X.shape[1],*y.shape[1:]])

    print('Finding coefficients')
    time.sleep(0.2)
    for i,j in tqdm(list(itertools.product(range(y.shape[1]), range(y.shape[2])))):
        p[:,i,j] = lstsq(X, y[:,i,j].transpose())[0]

    yhat = y.copy()
    print('Predicting SIC')
    time.sleep(0.2)

    yhat = np.einsum('tn,nij->tij',X,p)
    return p, yhat