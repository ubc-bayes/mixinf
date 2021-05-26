import autograd.numpy as np
from autograd import elementwise_grad as egrad
from autograd import grad

# banwidth of kernel
#k_gamma = 0.5
#k_gamma = 700


# kernel and first and second order derivatives
def get_kernel(k_gamma = 1):

    # define rbf kernel
    def kernel(t1, t2): return np.exp(-0.5 * np.sum((t1 - t2)**2, axis = -1) / k_gamma) # returns (N,1)

    # derivatives of the kernel
    def dk_x(t1, t2): return egrad(lambda t : kernel(t, t2))(t1) # returns (N,K)
    def dk_y(t1, t2): return egrad(lambda t : kernel(t1, t))(t2) # returns (N,K)
    def dk_xy(t1, t2):
        # returns (N,1)
        g = 0
        for d in range(t1.shape[1]):
            g += egrad(lambda t : dk_x(t1,t)[:, d])(t2)[:, d]

        return g

    return kernel, dk_x, dk_y, dk_xy
