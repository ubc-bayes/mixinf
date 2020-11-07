# suite of functions for doing variational mixture inference with normal kernels and sequential optimization

# preamble
import numpy as np
import scipy.stats as stats
import time, bisect
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import cProfile, pstats, io

###########
def grad(w, x, sd, H, qn, q, p, B, Y = None):
    """
    estimate the gradient of the objective function for current mixture q, defined by indices qn, at all choices of rho and x
    receives:
        - w, is a shape(n,) array of current weights
        - x is a shape(N,K) array of K-dim data points
        - sd is a shape(P,) array of kernel variances
        - H is a shape(N*P,2) array containing the dictionary of kernels (ie all possible combinations of mean and variance)
        - qn is a shape(n,) array containing the indices defining current mixture
        - q, p are functions (log probability densities) - q log mixture (e.g. output from q_gen) and p target log density
        - B is the number of MC samples to be used to estimate gradient
        - type is one of 'kl' or 'hellinger' - determimes which gradient to compute
        - tol is used if type = 'hellinger' to prevent gradient overflow

    output:
        - grad_w - shape(N*P, ) array, each entry the gradient of KL wrt weight of each basis function X sd
    """

    K = x.shape[1]
    NP = x.shape[0]*sd.shape[0]

    # the normalized sample is used to evaluate log pdf of kernels
    normalized_sample = np.random.randn(NP, B, K)
    sample = normalized_sample * sd[H[:, 1], np.newaxis, np.newaxis] + x[H[:, 0], np.newaxis, :]

    # first kernels: f(x) = 1/sigma phi(x - mu / sigma) to use std normal log pdf
    kernels = np.squeeze(norm_logpdf_3d(normalized_sample, loc = np.zeros((NP, K)), scale = np.ones(NP))) - np.log(sd[H[:, 1], np.newaxis])

    # now divergences
    grad = np.mean(kernels - p(sample), axis = -1)




    #Y = mixture_rvs(B, w, x[H[qn, 0:1], :], sd[H[qn, 1]]) # sample from current mixture
    #constant = np.mean(q(Y) - p(Y))
    #
    #for i in np.arange(NP):
    #    if K == 1:
    #        covm = (sd[H[i, 1]]**2).reshape(1, 1)
    #    else:
    #        covm = sd[H[i, 1]]**2 * np.eye(K)
    #
    #    X = np.random.multivariate_normal(mean = x[H[i, 0], :], cov = covm, size = B)
    #    grad_w = np.append(grad_w, np.mean(q(X) - p(X)) - constant)
    # end for


    return grad
###########


def w_grad(w, rho, x, q, p, B, Y = None, fixed_sampling = False):
    """
    estimate the gradient of the weights
    receives:
        - w, rho are shape(N,) arrays of weights, kernel variances
        - x is a shape(N,K) array of K-dim data points
        - q, p are functions (log probability densities) - q log mixture (e.g. output from q_gen) and p target log density
        - B is the number of MC samples to be used to estimate gradient
        - Y is a shape(N,B) array on which random deviates will be stored (if not None) to save memory
            * if fixed_sampling == True, this array is considered as the fixed sample and will be used to calculate the gradient
        - fixed_sampling is boolean. If False (i.e. continuous sampling), a new sample is genereated and used for gradient estimation.
            Else, a sample must be provided and will be used for gradient estimation.

    output:
        - grad_w- shape(N,K) array corresponding to the gradient of D(KL) w.r.t. w
    """


    if not fixed_sampling:
        # generate normal deviates if continuous sampling - row n is a sample of size B from qn
        Y = norm_random(loc = x, scale = rho, size = B)

        # estimate weight gradient
        grad_w = np.mean(q(Y) - np.squeeze(p(Y)), axis = -1)

    if fixed_sampling:
        N = w.shape[0]
        q_aux = q_gen(np.ones(N)/N, x, rho) # importance sampling distribution

        # estimate weight gradient
        grad_w = np.mean((q(Y) - np.squeeze(p(Y))) * np.exp(q(Y)) / np.maximum(np.exp(q_aux(Y)), 0.01), axis = -1)


    return grad_w
###########


###########
def q_gen(w, x, rho):
    """
    generate log density with current values w, x, rho
    x is a shape(N, K) array and w and rho are shape(N,) arrays
    the output is a vectorized function q
    """
    # define function that receives (n+1)darray y and returns q(y), an nd array
    # the n+1 th dimension accounts for multivariate data
    def q_out(y):
        # apply log sum exp trick
        ln = norm_logpdf(y, loc = x, scale = rho)
        target = np.log(w) + ln  # log sum wn exp(ln) = log sum exp(log wn + ln)
        max_value = np.max(target, axis = -1) # max within last axis
        exp_sum = np.exp(target - max_value[..., np.newaxis]).sum(axis = -1)
        return max_value + np.log(exp_sum)

    return q_out
###########


###########
def simplex_project(x):
    """
    project shape(N,) array x into the probabilistic simplex Delta(N-1)
    returns a shape (N,) array y

    code adapted from Duchi et al. (2008)
    """

    N = x.shape[0]
    mu = -np.sort(-x) # sort x in descending order
    rho_aux = mu - (1 / np.arange(1, N+1)) * (np.cumsum(mu) - 1) # build array to get rho
    rho = bisect.bisect_left(-rho_aux, 0) # first element grater than 0
    theta = (np.sum(mu[:rho]) - 1) / rho #
    x = np.maximum(x - theta, 0)

    return x
###########


###########
def mixture_rvs(size, w, x, rho):
    """
    draws a random sample of size size from the mixture defined by w, x, and rho
    x is a shape(N, K) array and w, rho are shape(N,) arrays
    returns a shape(size, K) array
    """
    N = x.shape[0]
    K = x.shape[1]

    inds = np.random.choice(N, size = size, p = w, replace = True) # indices that will be chosen

    #rand = np.random.multivariate_normal(mean = np.zeros(K), cov = np.eye(K), size = size) # sample from standard normal
    rand = np.random.randn(size, K) # sample from standard normal but more efficiently than as above

    # return scaled and translated random draws
    sigmas = rho[inds] # index std deviations for ease
    return rand * sigmas[:, np.newaxis] + x[inds, :]
###########


###########
def objective(p, q, w, x, rho, B = 1000, type = 'kl'):
    """
    estimate the objective function at current iteration
    - p, q are the target and mixture log densities, respectively
    - w, x, rho are shape(N,) arrays defining the current mixture
    - B is the number of MC samples used to estimate the objective function - has to be an integer
    - type is one of 'kl', 'hellinger', or 'l1' - determines which objective to compute

    returns a scalar
    """

    # generate random sample
    Y = mixture_rvs(size = B, w = w, x = x, rho = rho)
    pY = np.squeeze(p(Y))

    if type == 'kl': dist = np.mean(q(Y) - pY)

    if type == 'hellinger': dist = np.mean( (np.sqrt(np.exp(q(Y))) - np.sqrt(np.exp(pY)))**2 / np.maximum(1e-2, np.exp(q(Y))))

    if type == 'l1': dist = np.mean( np.abs( 1 - (np.exp(pY) / np.exp(q(Y))) ) )

    return dist
###########


###########
def norm_logpdf(x, loc = np.array([0]).reshape(1, 1), scale = np.array([1])):
    """
    evaluate isotropic normal logpdf at x with mean loc and sd scale

    - x is an (m+1)d array, where the last dimension accounts for multivariate x
        eg x[0, 0,..., 0, :] is the first observation and has shape K
    - loc is a shape(N, K) array
    - scale is a shape(N,). The covariance matrix is given by scale[i]**2 * np.diag(N) (ie Gaussians are isotropic)

    returns an md array with same shapes as x (except the last dimension)
    """
    K = x.shape[-1]

    return -0.5 * ((x[..., np.newaxis] - loc.T)**2).sum(axis = -2) / scale**2 - 0.5 * K *  np.log(2 * np.pi) - K * np.log(scale)
###########


###########
def norm_logpdf_3d(x, loc = np.array([0]).reshape(1, 1), scale = np.array([1])):
    """
    evaluate isotropic normal logpdf at x with mean loc and sd scale

    - x is a shape(N, B, K) array
    - loc is a shape(N, K) array
    - scale is a shape(N,). The covariance matrix is given by scale[i]**2 * np.diag(N) (ie Gaussians are isotropic)

    returns an md array with same shapes as x (except the last dimension)
    """
    K = x.shape[-1]

    return -0.5 * ((x - loc[:, np.newaxis, :])**2).sum(axis = -1) / scale[:, np.newaxis]**2 - 0.5 * K *  np.log(2 * np.pi) - K * np.log(scale[:, np.newaxis])
###########


###########
def norm_pdf(x, loc = np.array([0]).reshape(1, 1), scale = np.array([1])):
    """
    evaluate normal pdf at x with mean loc and sd scale
    if x is a shape(M,) array and both loc and scale are shape(N,) arrays, it returns an (N, M) array
    """
    return np.exp(norm_logpdf(x, loc = loc, scale = scale))
###########


###########
def norm_random(loc = np.array([0]).reshape(1, 1), scale = np.array([1]), size = 1):
    """
    generate size random numbers from N Normal distributions with means loc and isotropic covariance matrices scale

    - loc is a shape(N, K) array giving the N means of each K-dim Gaussian
    - scale is a shape(N,) array giving the N sd of the isotropic covariance matrices
    - size is an integer saying how many samples to generate

    returns a size(N, size, K) array
    """

    N = loc.shape[0]
    K = loc.shape[1]

    #rand = np.random.multivariate_normal(mean = np.zeros(K), cov = np.eye(K), size = (N, size)) # sample from standard normal
    rand = np.random.randn(N, size, K) # sample from standard normal but more efficiently than above
    return rand * scale[:, np.newaxis, np.newaxis] + loc[:, np.newaxis, :]
###########


###########
def kernel_init(p, x, rho, H, B = 500):
    """
    choose the first kernel to initialize optimization based on smallest divergence to target p

    - p is the target log density
    - x is a shape(N,K) array with the locations of the kernels
    - rho is a shape(P,) array with the variances
    - H is the dictionary containing all combinations of x and rho
    - B is the MC sample size to estimate the divergence (defaults to 500)

    out:
    - qn is a shape(1,) array with an integer between 1 and N*P which determines the dictionary index of the first kernel
    """

    K = x.shape[1]
    NP = H.shape[0]


    # the normalized sample is used to evaluate log pdf of kernels
    normalized_sample = np.random.randn(NP, B, K)
    sample = normalized_sample * rho[H[:, 1], np.newaxis, np.newaxis] + x[H[:, 0], np.newaxis, :]

    # first kernels: f(x) = 1/sigma phi(x - mu / sigma) to use std normal log pdf
    kernels = np.squeeze(norm_logpdf_3d(normalized_sample, loc = np.zeros((NP, K)), scale = np.ones(NP))) - np.log(rho[H[:, 1], np.newaxis])

    # now divergences
    divergences = np.mean(kernels - p(sample), axis = -1)

    #divergences = np.array([])
    #for i in np.arange(H.shape[0]):
    #    y = np.random.randn(B, K) * np.sqrt(rho[H[i, 1], np.newaxis]) + x[H[i, 0], :] # random sample from kernel
    #    divergences = np.append(divergences, np.mean(p(y) - norm_logpdf(y))) # estimate KL
    # end for

    return np.array([np.argmin(divergences)])

###########


###########
def w_opt(w, rho, x, p, maxiter = 500, B = 500, b = 0.01, tol = 1e-2, fixed_sampling = False, Y_aux = None, verbose = False):
    """
    jointly optimize weights for current mixture

    - w is a shape(n,) array with the current weights. The sgd will be initialized at [w, 0]
    - rho is a shape(n,) array with the current stadard deviations
    - x is a shape(n,K) array with the sample
    - p is the target log density

    optional arguments:
    - max number of iterations maxiter is an integer
    - MC sample size to estimate the gradient at each iteration B is an integer
    - schedule size to do step size b
    - tolerance to determine convergence tol
    - fixed_sampling is boolean. If False (i.e. continuous sampling), a new sample is genereated and used for gradient estimation.
        Else, a sample must be provided and will be used for gradient estimation.
    - Y_aux is a shape(B,K) array used for fixed sampling
    - verbose is a boolean

    returns a shape(n+1,) array with the optimal weights
    """


    w = np.append(w, 0)

    n = w.shape[0]
    Y_aux = np.ones((n, B)) # reduce memory storage by replacing entries in this array
    sgd_convergence = False
    w_step = 0

    #b = 0.1
    #B = 5000

    # if fixed sampling, create the sample from mixture with uniform weights
    if fixed_sampling: Y_aux = mixture_rvs(size = B, w = np.ones(n)/n, x = x, rho = rho)

    for l in range(maxiter):
        if verbose: print(':', end = ' ')
        # assess convergence
        if sgd_convergence: break

        # get gradient
        q = q_gen(w, x, rho)
        Dw = w_grad(w, rho, x, q, p, B, Y = Y_aux, fixed_sampling = fixed_sampling)

        # set step size
        #w_step = w_step = 0.9*w_step - (b/np.sqrt(l+1)) * Dw #momentum
        w_step = - (b/np.sqrt(l+1)) * Dw

        # update weights
        w += w_step
        w = simplex_project(w)

        # update convergence
        if np.linalg.norm(w_step) < tol: sgd_convergence = True
    # end for

    if verbose: print(':')

    return w
###########


###########
def nsvmi_grid(p, x, sd = np.array([1]), tol = 1e-2, maxiter = None, B = 500, fixed_sampling = False, trace = True, path = '', verbose = False, profiling = False):
    """
    receive target log density p and sample x
        - p is a function, and particularly a probability log density such as stat.norm.logpdf
        - x is a shape(N, K) numpy array

    optional arguments:
        - sd is a shape(P,) array with the different sd values to be considered
        - type is one of 'kl' or 'hellinger' to determine which target divergence to optimize
        - tolerance to determine convergence tol (double); defaults to 0.01
        - max number of components to add to mixture maxiter; if not specified will default to N, the sample size
        - number of MC samples for gradient estimation B (integer); defaults to 500
        - fixed_sampling is boolean. If False (i.e. continuous sampling), a new sample is genereated and used for gradient estimation.
            Else, a sample must be provided and will be used for gradient estimation.
        - silent is boolean. If False (default) a convergence message is printed. Else, no message is printed
        - trace is boolean. If True (default) it will estimate the objective function at each iteration
        - path is a string specifying the path to which trace plots should be saved (if trace = True)

    return optimal weights w, optimal kernel locations x and variances rho, target mixture log density q, and estimates of objective function
    w, x, and rho are shape(#iter,) numpy arrays and q is a function
    obj is a shape(#iter,) array with estimates of the objective function per iteration

    note that if tol is set to 0 then #iter = maxiter
    """

    start_time = time.time()


    # retrieve lengths
    N = x.shape[0]
    K = x.shape[1]
    P = sd.shape[0]

    # if not specified by user, maxiter defaults to sample size
    if maxiter is None: maxiter = N
    if maxiter > N*P: maxiter = N*P

    if verbose: print('Max iters: ' + str(maxiter))

    # dictionary
    if verbose: print('Creating kernel basis dictionary')
    H = np.array(np.meshgrid(np.arange(N), np.arange(P))).T.reshape(N*P, 2) # this merges both arrays including all combinations. column 0 contains indices for means

    # initialize values
    w = np.ones(1)
    convergence = False
    obj = np.array([])      # objective function array
    Y_aux = np.ones((N, B)) # reduce memory storage by replacing entries in this array

    # for now, randomly select first kernel
    #qn = np.random.choice(N*P, size = 1)          # kernel from the N*P available ones
    #ind = np.setdiff1d(range(N*P), qn)  # available indices

    if verbose: print('Choosing first kernel')
    qn = kernel_init(p, x, sd, H, B = 500) # choose kernel KL-closest to target as first approx
    if verbose: print('Selected index: ' + str(qn))
    if verbose: print('Location: ' + str(x[H[qn, 0], :]))
    if verbose: print('Std dev: ' + str(sd[H[qn, 1]]))

    if profiling:
        pr = cProfile.Profile()
        pr.enable()

    # do sequential procedure
    for k in np.arange(1, maxiter):
        if verbose: print('Iteration ' + str(k))
        #print(k)
        # assess convergence
        if convergence: break

        # get current mixture
        if verbose: print('Getting current mixture')
        x_curr = x[H[qn, 0], :]
        rho = sd[H[qn, 1]]
        q = q_gen(w = w, x = x_curr, rho = rho)

        # estimate gradients
        if verbose: print('Estimating gradients')
        grads = grad(w, x, sd, H, qn, q, p, B, Y = Y_aux)

        # select gradient with most negative entry
        if verbose: print('Selecting most negative gradient')
        smallest = np.argmin(grads)

        # update indices
        if verbose: print('Updating selected indices')
        qn = np.append(qn, smallest)
        #ind = np.setdiff1d(range(N*P), qn)
        if verbose: print('Selected indices: ' + str(qn))
        if verbose: print('Latest location: ' + str(x[H[qn[-1], 0], :]))
        if verbose: print('Latest std dev: ' + str(sd[H[qn[-1], 1]]**2))

        # optimize weights
        if verbose: print('Optimizing weights')
        w = w_opt(w = w, rho = sd[H[qn, 1]], x = x[H[qn, 0], :], p = p, maxiter = 1000, B = 5000, b = 0.1, tol = 1e-2, fixed_sampling = fixed_sampling, verbose = verbose)

        # remove small weights
        if verbose: print('Removing small weights')
        weight_tol = min(1e-2, 1/(4*w.shape[0])) # tol = 1% unless there are more than 100 basis functions, in which case it is 1/4 of uniform weights
        keep = np.nonzero(w > weight_tol)
        w = simplex_project(w[keep])
        qn = qn[keep]
        if verbose: print('Weights: ' + str(w))

        # estimate objective function
        if verbose: print('Estimating objective function')
        obj = np.append(obj, objective(p, q, w, x = x[H[qn, 0], :], rho = sd[H[qn, 1]], B = 100000, type = 'kl'))
        if verbose: print('Estimate = ' + str(obj))

        #update convergence if step size is small enough
        if k == 1:
            convergence = False
        elif  np.abs(obj[-1] - obj[-2]) < tol:
            #np.abs(obj[-1]) < tol
            #np.abs(obj[-1] - obj[-2])
            convergence = True

    # end for
    if profiling:
        pr.disable()
        print(pstats.Stats(pr).sort_stats('cumulative').print_stats())


    # create final mixture and output
    rho = sd[H[qn, 1]]
    x_out = x[H[qn, 0], :]
    q = q_gen(w, x_out, rho)

    # print message if not silent
    if verbose:
        time_elapsed = time.time() - start_time
        print(f"done in {k} iterations! time elapsed: {time_elapsed} seconds")

    # create trace plot if needed
    if trace:
        fig, ax1 = plt.subplots()
        plt.plot(np.arange(obj.shape[0]), obj, '-k')
        plt.xlabel('iteration')
        plt.ylabel('objective function')
        plt.title('trace plot of objective function')
        fig.tight_layout()
        plt.savefig(path + 'trace_N' + str(N) + '_' + 'K' + str(K) + '_' + str(time.time()) + '.pdf', dpi=900)

    return w, x_out, rho, q, obj
###########
