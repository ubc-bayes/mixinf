# suite of functions for doing locally-adapted boosting variational inference

# preamble
import numpy as np
import scipy.stats as stats
import cvxpy as cp
import time, bisect
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import cProfile, pstats, io


###################################
def mix_sample(size, y, T, w, logp, kernel_sampler):
    """
    sample from mixture of mcmc kernels

    inputs:
        - size number of samples
        - y kernel locations
        - T array with number of steps per kernel location
        - w array with location weights
        - logp target logdensity
        - kernel_sampler is a function that generates samples from the mixture kernels
    outputs:
        - shape(size,) array with the sample
    """

    N = y.shape[0]
    inds = np.random.choice(N, size = size, p = w, replace = True) # indices to sample from
    values, counts = np.unique(inds, return_counts = True) # sampled values with counts

    out = np.array([]) # init
    for i in range(values.shape[0]):
        # for each value, generate a sample of size counts[i]
        tmp_out = kernel_sampler(y = np.array([y[values[i]]]), T = np.array([T[values[i]]]), N = counts[i], logp = logp)

        # add to sample
        out = np.append(out, tmp_out)
    # end for

    return out
###################################


###################################
def LogSumExp(x):
    """
    LogSumExp trick for an array x
    """

    maxx = np.max(x)
    return maxx + np.log(np.sum(np.exp(x - maxx)))
###################################



###################################
def up_gen(kernel, sp, dk_x, dk_y, dk_xy):
    """
    generate the u_p function for ksd estimation

    inputs:
        - x, y arrays to evaluate at
        - k rkhs kernel
        - sp score of p
        - dk_x derivative of k wrt x
        - dk_y derivative of k wrt y
        - dk_xy trace of hessian of k
    outputs:
        - a function that takes two vectors as input and outputs u_p
    """

    def anon_up(x, y): return sp(x)*kernel(x, y)*sp(y) + sp(x)*dk_y(x, y) + sp(y)*dk_x(x, y) + dk_xy(x, y)

    return anon_up
###################################


###################################
def ksd(logp, y, T, w, up, kernel_sampler, B = 1000):
    """
    estimate ksd

    inputs:
        - p target logdensity
        - y kernel locations
        - T array with number of steps per kernel location
        - w array with location weights
        - up function to calculate expected value of
        - kernel_sampler is a function that generates samples from the mixture kernels
        - B number of MC samples
    outputs:
        - scalar, the estimated ksd
    """

    # generate samples
    X = mix_sample(2*B, y = y, T = T, w = w, logp = logp, kernel_sampler = kernel_sampler) # sample from mixture

    Y = X[-B:]
    X = X[:B]


    return up(X, Y).mean()
###################################



###################################
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
###################################


###################################
def plotting(y, T, w, logp, plot_path, iter_no, x_lower, x_upper, y_upper, kernel_sampler, N = 10000):
    """
    function that plots the target density and approximation, used in each iteration of the main routine

    inputs:
        - y array with kernel locations
        - T array with step sizes
        - w array with weights
        - logp target log density
        - plot_path string with plath to save figures in
        - iter_no integer with iteration number for title
        - x_lower, x_upper, y_upper plotting limits
        - kernel_sampler is a function that generates samples from the mixture kernels
        - N mixture sample size
    """

    # plot target density
    tt = np.linspace(x_lower, x_upper, 1000)
    zz = np.exp(logp(tt))
    plt.clf()
    plt.plot(tt, zz, '-k', label = 'target')

    # generate approximation
    kk = mix_sample(N, y, T, w, logp, kernel_sampler = kernel_sampler)

    yy = stats.gaussian_kde(kk, bw_method = 0.05).evaluate(tt)

    # plot approximation
    plt.plot(tt, yy, '--b', label = 'approximation')
    plt.hist(kk, label = 'approximation', density = True, bins = 30)
    plt.plot(y, np.zeros(y.shape[0]), 'ok')
    #plt.plot(y[argmin], np.zeros(1), 'or')
    plt.ylim(0, y_upper)
    plt.legend()
    plt.suptitle('l-bvi approximation to density')
    plt.title('iter: ' + str(iter_no))
    plt.savefig(plot_path + 'iter_' + str(iter_no) + '.jpg', dpi = 300)
###################################


###################################
def w_grad(up, logp, y, T, w, B, kernel_sampler):
    """
    calculate gradient of the KSD wrt to the weights

    inputs:
        - up function to calculate expected value of
        - logp target logdensity
        - y kernel locations
        - T array with number of steps per kernel location
        - w array with location weights
        - B number of MC samples
        - kernel_sampler is a function that generates samples from the mixture kernels
    outputs:
        - shape(y.shape[0],) array, the gradient
    """

    # init
    grad_w = np.zeros(y.shape[0])

    # sample from the mixture
    mix_X = mix_sample(2*B, y = y, T = T, w = w, logp = logp, kernel_sampler = kernel_sampler)
    mix_Y = mix_X[-B:]
    mix_X = mix_X[:B]

    X = kernel_sampler(y = y, T = T, N = 2*B, logp = logp)

    # sample from each kernel and define gradient
    for n in range(y.shape[0]):

        tmp_X = np.squeeze(X[:, n])
        tmp_Y = tmp_X[-B:]
        tmp_X = tmp_X[:B]

        # get gradient
        grad_w[n] = up(tmp_X, mix_X).mean() + up(mix_Y, tmp_Y).mean()
    # end for

    return grad_w
###################################


###################################
def weight_opt(logp, y, T, w, active, up, kernel_sampler, tol = 0.01, b = 0.1, B = 1000, maxiter = 1000, verbose = False, trace = False, tracepath = ''):
    """
    optimize weights via sgd

    inputs:
        - logp target density
        - y kernel locations
        - T array with number of steps per kernel location
        - w array with location weights
        - active array with number of active locations
        - up function to calculate expected value of
        - kernel_sampler is a function that generates samples from the mixture kernels
        - tol is the tolerance for convergence assessment
        - b is the optimization schedule
        - B number of MC samples
        - maxiter bounds the number of algo iterations
        - verbose is boolean indicating whether to print messages
        - trace is boolean indicating whether to print a trace plot of the objective function
        - tracepath is the path in which the trace plot is saved if generated
    outputs:
        - array with optimal weights of shape active.shape[0]
    """

    # if only one location is active, weight is 1
    if np.unique(active).shape[0] == 1: return np.array([1])


    # subset active locations
    y = y[active]
    T = T[active]
    w = w[active]
    #w = np.ones(w.shape[0]) / w.shape[0]
    w = w / w.sum()
    n = active.shape[0]

    # init algo
    convergence = False
    w_step = 0
    obj = np.array([])

    # run optimization
    for k in range(maxiter):

        if verbose: print(':', end = ' ') # to visualize number of iterations
        if convergence: break # assess convergence
        Dw = w_grad(up, logp, y, T, w, B, kernel_sampler = kernel_sampler) # get gradient
        w_step = 0.9*w_step - (b/np.sqrt(k+1)) * Dw # step size with momentum
        w_step = - (b/np.sqrt(k+1)) * Dw # step size without momentum
        w += w_step # update weight
        w = simplex_project(w) # project to simplex
        if np.linalg.norm(Dw) < tol: convergence = True # update convergence

        if trace: obj = np.append(obj, ksd(logp = logp, y = y, T = T, w = w, up = up, kernel_sampler = kernel_sampler, B = 5000))


    # end for

    if trace:
        plt.clf()
        plt.plot(1 + np.arange(obj.shape[0]), obj, '-k')
        plt.xlabel('iteration')
        plt.ylabel('kernelized stein discrepancy')
        plt.title('trace plot of ksd in weight optimization')
        plt.savefig(tracepath + str(np.sum(T) / t_increment) + '.jpg', dpi = 300)

    if verbose: print('Weights optimized in ' + str(k+1) + ' iterations')

    return w
###################################


###################################
def choose_kernel(up, logp, y, active, T, t_increment, t_max, w, B, kernel_sampler):
    """
    choose kernel to add to the mixture

    inputs:
        - up function to calculate expected value of
        - logp target logdensity
        - y kernel locations
        - active is an array with active locations
        - T array with number of steps per kernel location
        - t_increment integer with number of steps to increase chain by
        - t_max max number of steps allowed per chain
        - w array with location weights
        - B number of MC samples
        - kernel_sampler is a function that generates samples from the mixture kernels
    outputs:
        - integer with location that minimizes linear approximation
    """


    N = y.shape[0]
    grads = np.zeros(N)

    for n in range(N):

        # settings:
        tmp_active = np.setdiff1d(active, np.array([n])) # if chain is active, remove. else, do nothing

        # calculate exactly if this is the only active chain
        if tmp_active.size == 0:
            d0 = ksd(logp = logp, y = np.array([y[n]]), T = np.array([T[n]]), w = np.ones(1), up = up, kernel_sampler = kernel_sampler, B = B)
            d1 = ksd(logp = logp, y = np.array([y[n]]), T = np.array([T[n] + t_increment]), w = np.ones(1), up = up, kernel_sampler = kernel_sampler, B = B)
            grads[n] = d0 - d1 # exact decrease
            break
        #print('active chains: ' + str(tmp_active))
        #print('active locations: ' + str(y[tmp_active]))

        tmp_w = np.copy(w)
        tmp_w[n] = 0                   # the chain to be run is removed from mixture
        tmp_w = tmp_w[tmp_active]          # only active weights
        tmp_w = simplex_project(tmp_w) # and weights normalized
        #print('active weights: ' + str(tmp_w))

        tmp_T = np.copy(T)
        tmp_T[n] = tmp_T[n] + t_increment # increase number of steps in kernel n

        if tmp_T[n] > t_max:
            grads[n] = np.inf
        else:
            #tmp_T = tmp_T[tmp_active]
            #print('active steps: ' + str(tmp_T[tmp_active]))
            # generate samples
            X_mix = mix_sample(size = 4*B, y = y[tmp_active], T = tmp_T, w = tmp_w, logp = logp, kernel_sampler = kernel_sampler)
            X_kernel = kernel_sampler(y = np.array([y[n]]), T = np.array([tmp_T[n]]), N = 2*B, logp = logp)

            # estimate gradient
            grads[n] = up(X_mix[:B], X_kernel[:B]).mean() + up(X_kernel[B:2*B], X_mix[B:2*B]).mean() - 2*up(X_mix[2*B:3*B], X_mix[3*B:4*B]).mean()

    # end for
    print(grads)
    return np.argmin(grads)
###################################



###################################
def lbvi(y, logp, t_increment, t_max, up, kernel_sampler, w_maxiters = None, w_schedule = None, B = 1000, maxiter = 100, tol = 0.001, weight_max = 10, verbose = False, plot = True, plot_path = 'plots', trace = False):
    """
    locally-adapted boosting variational inference main routine
    given a sample and a target, find the mixture of user-defined kernels that best approximates the target

    inputs:
        - y array of kernel locations (sample)
        - logp is a function, the target log density
        - t_increment integer with number of steps to increase chain by
        - t_max integer with max number of steps allowed per chain
        - up function to calculate expected value of when estimating ksd
        - kernel_sampler is a function that generates samples from the mixture kernels
        - w_maxiters is a function that receives the iteration number and a boolean indicating whether opt is long or not and outputs the max number of iterations for weight optimization
        - w_schedule is a function that receives the iteration number and outputs the step size for the weight optimization
        - B number of MC samples for estimating ksd and gradients
        - maxiter is an integer with the max the number of algo iterations
        - tol is a float with the tolerance below which the algorithm breaks the loop and stops
        - weight_max is an integer indicating max number of iterations without weight optimization (if no new kernels are added to mixture)
        - verbose is boolean indicating whether to print messages
        - trace is boolean indicating whether to print a trace plot of the objective function
        - tracepath is the path in which the trace plot is saved if generated
    outputs:
        - w, T are shape(y.shape[0], ) arrays with the sample, the weights, and the steps sizes
        - obj is an array with the value of the objective function at each iteration
    """

    N = y.shape[0]

    if w_maxiters is None:
        # define sgd maxiter function
        def w_maxiters(k, long_opt = False):
            if k == 0: return 350
            if long_opt: return 100
            return 50

    if w_schedule is None:
        # define schedule function
        def w_schedule(k):
            if k == 0: return 0.005
            return 0.000001

    # init values
    w = np.zeros(N)
    T = np.zeros(N, dtype = np.intc)
    convergence = False
    obj = np.array([])
    weight_opt_counter = 0
    long_opt = False

    # init mixture
    if verbose: print('choosing first mixture')
    tmp_ksd = np.zeros(N)
    for n in range(N):
        tmp_ksd[n] = ksd(logp = logp, y = np.array([y[n]]), T = np.array([t_increment]), w = np.ones(1), up = up, kernel_sampler = kernel_sampler, B = B)
        # end for

    argmin = np.argmin(tmp_ksd) # ksd minimizer
    if verbose: print('first element: ' + str(y[argmin]))
    w[argmin] = 1 # update weight
    T[argmin] = t_increment # update steps
    active = np.array([argmin]) # update active locations, kernel_sampler
    if verbose: print('number of steps: ' + str(T))
    obj = np.append(obj, tmp_ksd[argmin]) # update objective


    # plot initial approximation
    if verbose: print('plotting')
    plotting(y, T, w, logp, plot_path, iter_no = 0, x_lower = -6, x_upper = 6, y_upper = 1.5, kernel_sampler = kernel_sampler, N = 10000)

    if verbose: print()


    for iter_no in range(maxiter):

        if verbose: print('iteration ' + str(iter_no + 1))

        if verbose: print('assessing convergence')
        if convergence: break


        if verbose: print('choosing next step')
        argmin = choose_kernel(up, logp, y, active, T, t_increment, t_max, w, B = B, kernel_sampler = kernel_sampler)
        if verbose: print('chosen element: ' + str(y[argmin]))

        # update steps
        T[argmin] = T[argmin] + t_increment
        if verbose: print('number of steps: ' + str(T))


        # update active set
        if argmin not in active:
            active = np.append(active, argmin) # update active locations
            update_weights = True
            weight_opt_counter = 0
            long_opt = True
        elif weight_opt_counter > weight_max:
            update_weights = True
            weight_opt_counter  = 0
            long_opt = False
        else:
            update_weights = False
            weight_opt_counter += 1


        # update weights
        if update_weights:
            if verbose: print('updating weights')
            w[active] = weight_opt(logp, y, T, w, active, up, kernel_sampler = kernel_sampler, tol = 0.1, b = w_schedule(iter_no), B = B, maxiter = w_maxiters(iter_no, long_opt), verbose = verbose, trace = trace, tracepath = plot_path + 'weight_trace/')
            if verbose: print('weights: ' + str(w))
        else:
            if verbose: print('not updating weights')

        # estimate objective
        if verbose: print('estimating objective function')
        obj = np.append(obj, ksd(logp = logp, y = y, T = T, w = w, up = up, kernel_sampler = kernel_sampler, B = B))
        if verbose: print('objective: ' + str(obj[-1]))


        if verbose: print('updating convergence')
        if np.abs(obj[-1]) < tol: convergence = True


        if verbose: print('plotting')
        plotting(y, T, w, logp, plot_path, iter_no = iter_no + 1, x_lower = -6, x_upper = 6, y_upper = 1.5, kernel_sampler = kernel_sampler, N = 10000)

        if verbose: print()
        # end for

    if verbose: print('done!')
    if verbose: print('sample: ' + str(y))
    if verbose: print('weights: ' + str(w))
    if verbose: print('steps: ' + str(T))
    if verbose: print('ksd: ' + str(obj[-1]))

    return w, T, obj
