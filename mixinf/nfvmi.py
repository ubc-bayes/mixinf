# suite of functions for doing variational mixture inference with normal kernels and full optimization

# preamble
import numpy as np
import scipy.stats as stats
import time, bisect
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import cProfile, pstats, io

###########
def grad(w, rho, x, q, p, B, type = 'kl', tol = 1e-2, Y = None):
    """
    estimate the gradient of both the weights and the kernel variances
    receives:
        - w, rho are shape(N*P,) arrays of weights, extended kernel variances
        - x is a shape(N*P,K) array of extended K-dim data points
        - q, p are functions (log probability densities) - q log mixture (e.g. output from q_gen) and p target log density
        - B is the number of MC samples to be used to estimate gradient
        - type is one of 'kl' or 'hellinger' - determines which gradient to compute
        - tol is used if type = 'hellinger' to prevent gradient overflow

    output:
        - grad_w, grad_rho - shape(N,K) arrays corresponding to the gradients of D(type) w.r.t. w and rho, respectively
    """

    # generate normal deviates - row n is a sample of size B from qn
    Y = norm_random(loc = x, scale = rho, size = B)
    K = x.shape[1]

    if type == 'kl':
        # estimate weight gradient
        grad_w = np.mean(q(Y) - p(Y), axis = -1)

        # estimate kernel sd gradient
        grad_rho = np.mean((w / rho)[:, np.newaxis] * (q(Y) - p(Y)) * ( ((Y - x[:, np.newaxis, :])**2).sum(axis = -1) / rho[:, np.newaxis]**2 - K), axis = -1)

    if type == 'hellinger':
        # estimate weight gradient
        grad_w = - np.mean(np.sqrt(np.exp(p(Y)) / np.maximum(tol, np.exp(q(Y)))), axis = -1)

        # estimate kernel sd gradient
        grad_rho = - np.mean((w / rho)[:, np.newaxis] * np.sqrt(np.exp(p(Y)) / np.maximum(tol, np.exp(q(Y)))) * ( ((Y - x[:, np.newaxis, :])**2).sum(axis = -1) / rho[:, np.newaxis]**2 - K), axis = -1)


    return grad_w, grad_rho
###########


###########
def grad_w(w, rho, x, q, p, B, type = 'kl', tol = 1e-2, Y = None):
    """
    identical to grad but without calculating gradient of variance
    estimate the gradient of the weights only, for grid variance optimization
    receives:
        - w, rho are shape(N*P,) arrays of weights, extended kernel variances
        - x is a shape(N*P,K) array of extended K-dim data points
        - q, p are functions (log probability densities) - q log mixture (e.g. output from q_gen) and p target log density
        - B is the number of MC samples to be used to estimate gradient
        - type is one of 'kl' or 'hellinger' - determines which gradient to compute
        - tol is used if type = 'hellinger' to prevent gradient overflow

    output:
        - grad_w - shape(N,K) array corresponding to the gradients of D(type) w.r.t. w
    """

    # generate normal deviates - row n is a sample of size B from qn
    Y = norm_random(loc = x, scale = rho, size = B)
    K = x.shape[1]

    if type == 'kl': grad_w = np.mean(q(Y) - p(Y), axis = -1)

    if type == 'hellinger': grad_w = - np.mean(np.sqrt(np.exp(p(Y)) / np.maximum(tol, np.exp(q(Y)))), axis = -1)


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
    return rand * np.sqrt(sigmas[:, np.newaxis]) + x[inds]
###########


###########
def objective(p, q, w, x, rho, B = 1000, type = 'kl'):
    """
    estimate the objective function at current iteration
    - p, q are the target and mixture log densities, respectively
    - w, x, rho are shape(N,) arrays defining the current mixture
    - B is the number of MC samples used to estimate the objective function - has to be an integer
    - type is one of 'kl', 'hellinger', or 'l1' - determimes which objective to compute

    returns a scalar
    """

    # generate random sample
    Y = mixture_rvs(size = B, w = w, x = x, rho = rho)

    pY = np.squeeze(p(Y))

    if type == 'kl': dist = np.mean(q(Y) - pY)

    if type == 'hellinger': dist = np.mean( (np.sqrt(np.exp(q(Y))) - np.sqrt(np.exp(pY)))**2 / np.maximum(1e-2, np.exp(q(Y))))

    if type == 'l1': dist = np.minimum(2, np.mean( np.abs( 1 - (np.exp(pY) / np.exp(q(Y))) ) ))

    return dist
###########


###########
def norm_logpdf(x, loc = np.array([0]).reshape(1, 1), scale = np.array([1])):
    """
    evaluate isotropic normal logpdf at x with mean loc and sd scale
    x is an (m+1)d array, where the last dimension accounts for multivariate x
        eg x[0, 0,..., 0, :] is the first observation and has shape K
    loc is a shape(N, K) array
    scale is a shape(N,) array. The covariance matrix is given by scale[i]**2 * np.diag(N) (ie Gaussians are isotropic)

    returns an md array with same shapes as x (except the last dimension)
    """
    K = x.shape[-1]

    return -0.5 * ((x[..., np.newaxis] - loc.T)**2).sum(axis = -2) / scale**2 - 0.5 * K *  np.log(2 * np.pi) - K * np.log(scale)
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
    return rand * np.sqrt(scale[:, np.newaxis, np.newaxis]) + loc[:, np.newaxis, :]
###########



###########
def w_init(x, rho, p):
    """
    initialize weights for optimization routine

    - x is a shape(N,K) array with the location of the basis kernels
    - rho is a shape(N,) array with the variance of the basis kernels
    - p is the (possibly unnormalized) target log density

    returns a size(N,) array that lives in the N simplex
    """

    N = x.shape[0]
    w = p(x)

    return w / w.sum()
###########


###########
def nfvmi_grid(p, x, rho = np.array([1]), type = 'kl', tol = 1e-3, maxiter = 1e3, B = 500, b = 0.1, trace = True, path = '', verbose = False, profiling = False):
    """
    receive target log density p and sample x
        - p is a function, and particularly a probability log density such as stat.norm.pdf
        - x is a shape(N,) numpy array

    optional arguments:
        - rho is a shape(P,) array with the possible options for the variance
        - type is one of 'kl' or 'hellinger' to depermine which target distance to optimize
        - tolerance to determine convergence tol (double); defaults to 0.001
        - max number of SGD iterations maxiter (integer); defaults to 1000
        - number of MC samples for gradient estimation B (integer); defaults to 500
        - step size for SGD b (double); defaults to 1
        - silent is boolean. If False (default) a convergence message is printed. Else, no message is printed
        - trace is boolean. If True (default) it will estimate the objective function at each iteration

    return optimal weights w, optimal kernel basis x, optimal kernel variances rho, and target mixture log density q
    - x_extended is a shape(N*P, K) array
    - w and rho_extended are shape(N*P,) numpy arrays
    - q is a function
    - obj is a shape(#iter,) array with estimates of the objective function per iteration
    """

    start_time = time.time()
    maxiter += 1

    # retrieve lengths
    N = x.shape[0]
    K = x.shape[1]
    P = rho.shape[0]

    if verbose: print('Max iters: ' + str(maxiter-1))

    # dictionary
    if verbose: print('Creating kernel basis dictionary')
    H = np.array(np.meshgrid(np.arange(N), np.arange(P))).T.reshape(N*P, 2) # this merges both arrays including all combinations. column 0 contains indices for means
    x_extended = x[H[:, 0], ...]
    rho_extended = rho[H[:, 1]]

    # initialize values
    if verbose: print('Initializing values')
    w = np.ones(N*P) / (N*P)      # max entropy if wn = 1/N
    w = w_init(x_extended, rho_extended, p)
    convergence = False
    obj = np.array([])      # objective function array
    Y_aux = np.ones((N*P, B)) # reduce memory storage by replacing entries in this array

    # momentum variables:
    w_step = 0
    rho_step = 0

    if profiling:
        pr = cProfile.Profile()
        pr.enable()

    # do sgd
    for k in np.arange(1, maxiter):
        if verbose: print('Iteration ' + str(k))

        # assess convergence
        if convergence: break

        # get current q and estimate gradient
        if verbose: print('Getting current mixture')
        q = q_gen(w, x_extended, rho_extended)

        if verbose: print('Estimating gradient of weights')
        Dw = grad_w(w, rho_extended, x_extended, q, p, B, type = type, Y = Y_aux)

        if trace:
            # evaluate objective function
            if verbose: print('Estimating objective function')
            obj = np.append(obj, objective(p, q, w, x_extended, rho_extended, B = 1000, type = type))
            if verbose: print('Latest estimate: ' + str(obj[-1]))

        # get step sizes with momentum
        if verbose: print('Calculating step size')
        w_step = 0.9*w_step - (b/np.sqrt(k)) * Dw
        #if verbose: print('Weight step size: ' + str(w_step))

        # update weight values
        if verbose: print('Updating weights')
        w += w_step

        # project w to simplex
        w = simplex_project(w)
        if verbose: print('Proportion of zero weights: ' + str(np.count_nonzero(w < 1e-3) / w.shape[0]))


        #update convergence if step size is small enough
        if np.linalg.norm(w_step) < tol: convergence = True

    # end for
    if profiling:
        pr.disable()
        print(pstats.Stats(pr).sort_stats('cumulative').print_stats())


    # create final mixture
    q = q_gen(w, x_extended, rho_extended)

    # print message if not silent
    if verbose:
        time_elapsed = time.time() - start_time
        print(f"done in {k} iterations! time elapsed: {time_elapsed} seconds")
        print(f"weight step size: {np.linalg.norm(w_step)}")

    # create trace plot if needed
    if trace:
        fig, ax1 = plt.subplots()
        plt.plot(np.arange(obj.shape[0]), obj, '-k')
        plt.xlabel('iteration')
        plt.ylabel('objective function')
        plt.title('trace plot of objective function')
        fig.tight_layout()
        plt.savefig(path + 'trace_N' + str(N) + '_' + 'K' + str(K) + '_' + str(time.time()) + '.pdf', dpi=900)

    return w, x_extended, rho_extended, q, obj
###########




###########
def nfvmi(p, x, type = 'kl', tol = 1e-3, maxiter = 1e3, B = 500, b = 0.1, trace = True, path = '', verbose = False, profiling = False):
    """
    receive target log density p and sample x
        - p is a function, and particularly a probability log density such as stat.norm.pdf
        - x is a shape(N,) numpy array

    optional arguments:
        - type is one of 'kl' or 'hellinger' to depermine which target distance to optimize
        - tolerance to determine convergence tol (double); defaults to 0.001
        - max number of SGD iterations maxiter (integer); defaults to 1000
        - number of MC samples for gradient estimation B (integer); defaults to 500
        - step size for SGD b (double); defaults to 1
        - silent is boolean. If False (default) a convergence message is printed. Else, no message is printed
        - trace is boolean. If True (default) it will estimate the objective function at each iteration

    return optimal weights w, optimal kernel variances rho, and target mixture log density q
    w and rho are shape(N,) numpy arrays and q is a function
    obj is a shape(#iter,) array with estimates of the objective function per iteration
    """

    start_time = time.time()
    maxiter += 1

    # retrieve lengths
    N = x.shape[0]
    K = x.shape[1]

    # initialize values
    w = np.ones(N) / N      # max entropy if wn = 1/N
    rho = 2*np.ones(N)        # arbitrary choice
    convergence = False
    obj = np.array([])      # objective function array
    Y_aux = np.ones((N, B)) # reduce memory storage by replacing entries in this array
    # momentum variables:
    w_step = 0
    rho_step = 0

    # do sgd
    for k in np.arange(1, maxiter):
        # assess convergence
        if convergence: break

        # get current q and estimate gradient
        q = q_gen(w, x, rho)
        Dw, Drho = grad(w, rho, x, q, p, B, type = type, Y = Y_aux)

        if trace:
            # evaluate objective function
            obj = np.append(obj, objective(p, q, w, x, rho, B = 1000, type = type))

        # get step sizes with momentum
        w_step = 0.9*w_step - (b/np.sqrt(k)) * Dw
        rho_step = 0.9*rho_step - (b/np.sqrt(k)) * Drho

        # update values
        w += w_step
        rho += rho_step

        # project w to simplex and rho to R+
        w = simplex_project(w)
        rho = np.maximum(1e-2, rho)


        #update convergence if step size is small enough
        if (np.linalg.norm(w_step) < tol) & (np.linalg.norm(rho_step) < tol):
            convergence = True

    # end for

    # create final mixture
    q = q_gen(w, x, rho)

    # print message if not silent
    if not silent:
        time_elapsed = time.time() - start_time
        print(f"done in {k} iterations! time elapsed: {time_elapsed} seconds")

        w_norm, rho_norm = np.linalg.norm(w_step), np.linalg.norm(rho_step)
        print(f"weight step size: {w_norm} \nrho step size: {rho_norm}")

    # create trace plot if needed
    if trace:
        fig, ax1 = plt.subplots()
        plt.plot(np.arange(obj.shape[0]), obj, '-k')
        plt.xlabel('iteration')
        plt.ylabel('objective function')
        plt.title('trace plot of objective function')
        fig.tight_layout()
        plt.savefig(path + 'trace_N' + str(N) + '_' + 'K' + str(K) + '_' + str(time.time()) + '.pdf', dpi=900)

    return w, rho, q, obj
###########
