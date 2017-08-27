import numpy as np

x=np.array([1,2,3,4])

it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

grad = np.zeros(x.shape)

grad[0] = np.array([1,2,3])

while not it.finished:

    # evaluate function at x+h
    ix = it.multi_index
    print (ix)

    it.iternext()