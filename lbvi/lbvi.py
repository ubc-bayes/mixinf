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
def mix_sample(size, y, T, w, logp, kernel_sampler, verbose = False):
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
        - shape(size,K) array with the sample
    """

    N = y.shape[0]
    K = y.shape[1]
    y = y.reshape(N,K)
    inds = np.random.choice(N, size = size, p = w, replace = True) # indices to sample from
    values, counts = np.unique(inds, return_counts = True) # sampled values with counts

    if verbose:
        print()
        print('values: ' + str(values))
        print('counts: ' + str(counts))
        print('sample: ' + str(np.squeeze(y)))
        print('weights: ' + str(w))
        print('sample via values: ' + str(np.squeeze(y[values,:])))
        print('empirical weights: ' + str(counts / counts.sum()))
        print()

    out = np.empty((1, K)) # init
    for i in range(values.shape[0]):

        # for each value, generate a sample of size counts[i]
        if verbose: print('sampling from kernel centered at ' + str(np.squeeze(y[values[i],:])) + ' with steps ' + str(T[values[i]]))
        if verbose: print('empirical weight: ' + str(counts[i] / counts.sum()))
        tmp_out = kernel_sampler(y = np.array(y[values[i], :]).reshape(1, K), T = np.array([T[values[i]]]), S = counts[i], logp = logp).reshape(counts[i], K)
        if verbose: print('mean from this sample: ' + str(tmp_out.mean()))

        # add to sample
        out = np.concatenate((out, tmp_out))
    # end for
    if verbose:
        print()
        print('sample mean: ' + str(out.mean()))
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
        - x, y shape(N, K) arrays to evaluate at
        - k rkhs kernel
        - sp score of p
        - dk_x derivative of k wrt x
        - dk_y derivative of k wrt y
        - dk_xy trace of hessian of k
    outputs:
        - a function that takes two vectors as input and outputs u_p
    """

    def anon_up(x, y):
        # x, y shape(N,K)
        # out shape(N,1)

        term1 = np.squeeze(np.matmul(sp(x)[:,np.newaxis,:], sp(y)[:,:,np.newaxis])) * kernel(x, y)
        term2 = np.squeeze(np.matmul(sp(x)[:,np.newaxis,:], dk_y(x, y)[:,:,np.newaxis]))
        term3 = np.squeeze(np.matmul(sp(y)[:,np.newaxis,:], dk_x(x, y)[:,:,np.newaxis]))
        term4 = dk_xy(x, y)
        return term1 + term2 + term3 + term4

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

    Y = X[-B:, :]
    X = X[:B, :]

    return np.abs(up(X, Y).mean())
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
    rho = bisect.bisect_left(-rho_aux, 0) # first element greater than 0
    theta = (np.sum(mu[:rho]) - 1) / rho #
    x = np.maximum(x - theta, 0)

    return x
###################################


###################################
def plotting(y, T, w, logp, plot_path, iter_no, kernel_sampler = None, plt_lims = None, N = 10000):
    """
    function that plots the target density and approximation, used in each iteration of the main routine

    inputs:
        - y array with kernel locations
        - T array with step sizes
        - w array with weights
        - logp target log density
        - plot_path string with plath to save figures in
        - iter_no integer with iteration number for title
        - kernel_sampler is a function that generates samples from the mixture kernels (if None, the sample is plotted)
        - plt_lims is an array with the plotting limits (xinf, xsup, yinf, ysup)
        - N mixture sample size
    """
    plt.clf()

    # univariate data plotting
    if y.shape[1] == 1:
        # get plotting limits
        x_lower = plt_lims[0]
        x_upper = plt_lims[1]
        y_upper = plt_lims[3]

        # plot target density
        tt = np.linspace(x_lower, x_upper, 1000)
        zz = np.exp(logp(tt[:, np.newaxis]))
        plt.plot(tt, zz, '-k', label = 'target')

        if kernel_sampler is None:
            plt.hist(y, label = '', density = True, alpha = 0.5, facecolor = '#39558CFF', edgecolor='black')
        else:
            # generate and plot approximation
            lbvi_sample = np.squeeze(mix_sample(N, y, T, w, logp, kernel_sampler = kernel_sampler, verbose = False))
            plt.hist(lbvi_sample, label = 'LBVI', density = True, bins = 75, alpha = 0.5, facecolor = 'blue', edgecolor='black')
            plt.plot(np.squeeze(y), np.zeros(y.shape[0]), 'ok')

        # beautify and save plot
        plt.ylim(0, y_upper)
        plt.xlim(x_lower, x_upper)
        plt.legend()
        #plt.suptitle('l-bvi approximation to density')
        #plt.title('iter: ' + str(iter_no))
        plt.savefig(plot_path + 'iter_' + str(iter_no) + '.jpg', dpi = 300)

    # bivariate dataplotting
    if y.shape[1] == 2:
        # get plotting limits
        x_lower = plt_lims[0]
        x_upper = plt_lims[1]
        y_lower = plt_lims[2]
        y_upper = plt_lims[3]

        # plot target density
        nn = 100
        xx = np.linspace(x_lower, x_upper, nn)
        yy = np.linspace(y_lower, y_upper, nn)
        tt = np.array(np.meshgrid(xx, yy)).T.reshape(nn**2, 2)
        lp = logp(tt).reshape(nn, nn).T
        cp = plt.contour(xx, yy, np.exp(lp), colors = 'black')
        hcp,_ = cp.legend_elements()
        hcps = [hcp[0]]
        legends = ['p(x)']

        if kernel_sampler is None:
            plt.scatter(y[:,0], y[:,1], marker='.', c='k', alpha = 0.2, label = '')
        else:
            # generate and plot approximation
            lbvi_sample = mix_sample(N, y, T, w, logp, kernel_sampler = kernel_sampler)
            #plt.scatter(kk[:,0], kk[:,1], marker='.', c='k', alpha = 0.2, label = 'approximation')
            lbvi_kde = stats.gaussian_kde(lbvi_sample.T, bw_method = 0.05).evaluate(tt.T).reshape(nn, nn).T
            cp_lbvi = plt.contour(xx, yy, lbvi_kde, levels = Levels, colors = '#39558CFF')
            hcp_lbvi,_ = cp_lbvi.legend_elements()
            hcps.append(hcp_lbvi[0])
            legends.append('LBVI')

        # beautify and save plot
        plt.ylim(y_lower, y_upper)
        plt.xlim(x_lower, x_upper)
        #plt.suptitle('l-bvi approximation to density')
        # assign plot title
        #if kernel_sampler is None:
        #    plt.title('initial sample')
        #else:
        #    plt.title('iter: ' + str(iter_no))
        plt.savefig(plot_path + 'iter_' + str(iter_no) + '.jpg', dpi = 300)

###################################


###################################
def w_grad(up, logp, y, T, w, B, kernel_sampler):
    """
    calculate gradient of the KSD wrt to the weights

    inputs:
        - up function to calculate expected value of
        - logp target logdensity
        - y shape(N,K) kernel locations
        - T array with number of steps per kernel location
        - w array with location weights
        - B number of MC samples
        - kernel_sampler is a function that generates samples from the mixture kernels
    outputs:
        - shape(y.shape[0],) array, the gradient
    """

    N = y.shape[0]
    K = y.shape[1]

    # init
    grad_w = np.zeros(N)

    # sample from the mixture
    mix_X = mix_sample(2*B, y = y, T = T, w = w, logp = logp, kernel_sampler = kernel_sampler)
    mix_Y = mix_X[-B:]
    mix_X = mix_X[:B]

    X = kernel_sampler(y = y, T = T, S = 2*B, logp = logp)

    # sample from each kernel and define gradient
    for n in range(y.shape[0]):
        tmp_X = X[:,n,:]
        tmp_Y = tmp_X[-B:,:]
        tmp_X = tmp_X[:B,:]

        # get gradient
        grad_w[n] = up(tmp_X, mix_X).mean() + up(mix_Y, tmp_Y).mean()
    # end for

    return grad_w
###################################


###################################
def weight_opt(logp, y, T, w, active, up, kernel_sampler, t_increment, tol = 0.001, b = 0.1, B = 1000, maxiter = 1000, verbose = False, trace = False, tracepath = ''):
    """
    optimize weights via sgd

    inputs:
        - logp target density
        - y shape(N,K) kernel locations
        - T array with number of steps per kernel location
        - w array with location weights
        - active array with number of active locations
        - up function to calculate expected value of
        - kernel_sampler is a function that generates samples from the mixture kernels
        - t_inrement is an integer with the number of increments per step (used only for plotting)
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
    y = y[active,:]
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

        if verbose: print(str(k+1) + '/' + str(maxiter), end = '\r') # to visualize number of iterations
        if convergence: break # assess convergence
        Dw = w_grad(up, logp, y, T, w, B, kernel_sampler = kernel_sampler) # get gradient
        #w_step = 0.9*w_step - (b/np.sqrt(k+1)) * Dw # step size with momentum
        w_step = - (b/np.sqrt(k+1)) * Dw # step size without momentum
        w += w_step # update weight
        w = simplex_project(w) # project to simplex
        if np.linalg.norm(Dw) < tol: convergence = True # update convergence

        if trace: obj = np.append(obj, ksd(logp = logp, y = y, T = T, w = w, up = up, kernel_sampler = kernel_sampler, B = 10000))


    # end for

    if trace:
        plt.clf()
        plt.plot(1 + np.arange(obj.shape[0]), obj, '-k')
        plt.xlabel('iteration')
        plt.ylabel('kernelized stein discrepancy')
        plt.title('trace plot of ksd in weight optimization')
        plt.savefig(tracepath + str(np.sum(T) / t_increment) + '.jpg', dpi = 300)

    if verbose:
        print('weights optimized in ' + str(k+1) + ' iterations')

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
            d0 = ksd(logp = logp, y = np.array([y[n]]), T = np.array([T[n]]), w = np.ones(1), up = up, kernel_sampler = kernel_sampler, B = 100000)
            d1 = ksd(logp = logp, y = np.array([y[n]]), T = np.array([T[n] + t_increment]), w = np.ones(1), up = up, kernel_sampler = kernel_sampler, B = 100000)
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
            X_kernel = kernel_sampler(y = np.array([y[n]]), T = np.array([tmp_T[n]]), S = 2*B, logp = logp)[:,0,:]

            # estimate gradient
            grads[n] = up(X_mix[:B,:], X_kernel[:B,:]).mean() + up(X_kernel[B:2*B,:], X_mix[B:2*B,:]).mean() - 2*up(X_mix[2*B:3*B,:], X_mix[3*B:4*B,:]).mean()

    # end for
    #print('sample: ' + str(np.squeeze(y)))
    #print('gradients: ' + str(grads))
    return np.argmin(grads)
###################################



###################################
def lbvi(y, logp, t_increment, t_max, up, kernel_sampler, w_maxiters = None, w_schedule = None, B = 1000, maxiter = 100, tol = 0.001, stop_up = None, weight_max = 20, verbose = False, plot = True, plt_lims = None, plot_path = 'plots', trace = False):
    """
    locally-adapted boosting variational inference main routine
    given a sample and a target, find the mixture of user-defined kernels that best approximates the target

    inputs:
        - y shape(N,K) array of kernel locations (sample)
        - logp is a function, the target log density
        - t_increment integer with number of steps to increase chain by
        - t_max integer with max number of steps allowed per chaikernel_samplern
        - up function to calculate expected value of when estimating ksd
        - kernel_sampler is a function that generates samples from the mixture kernels
        - w_maxiters is a function that receives the iteration number and a boolean indicating whether opt is long or not and outputs the max number of iterations for weight optimization
        - w_schedule is a function that receives the iteration number and outputs the step size for the weight optimization
        - B number of MC samples for estimating ksd and gradients
        - maxiter is an integer with the max the number of algo iterations
        - tol is a float with the tolerance below which the algorithm breaks the loop and stops
        - stop_up indicates the stopping criterion. If None, the ksd provided will be used. Else, provide an auxiliary up function, which will be used to build a surrogate ksd to determine convergence
        - weight_max is an integer indicating max number of iterations without weight optimization (if no new kernels are added to mixture)
        - verbose is boolean indicating whether to print messages
        - plot is boolean indicating whether to generate plots of the approximation at each iteration (only supported for uni and bivariate data)
        - plt_lims is an array with the plotting limits (xinf, xsup, yinf, ysup)
        - trace is boolean indicating whether to print a trace plot of the objective function
        - tracepath is the path in which the trace plot is saved if generated
        - w, T are shape(y.shape[0], ) arrays with the sample, the weights, and the steps sizes
        - obj is an array with the value of the objective function at each iteration
    """
    if verbose:
        print('running locally-adapted boosting variational inference')
        print()

    t0 = time.perf_counter()
    N = y.shape[0]
    K = y.shape[1]
    if stop_up is None:
        stop_up = up
    else:
        if verbose:
            print('using surrogate up for convergence assessment')
            print()

    if K > 2: plot = False # no plotting beyond bivariate data

    if w_maxiters is None:
        # define sgd maxiter function
        def w_maxiters(k, long_opt = False):
            if k == 0: return 350
            if long_opt: return 50
            return 50

    if w_schedule is None:
        # define schedule function
        def w_schedule(k):
            if k == 0: return 0.005
            return 0.000001

    # plot initial sample
    if plot: plotting(y, T = 0, w = 0, logp = logp, plot_path = plot_path, iter_no = -1, kernel_sampler = None, plt_lims = plt_lims, N = 10000)

    # init values
    w = np.zeros(N)
    T = np.zeros(N, dtype = np.intc)
    convergence = False
    weight_opt_counter = 0
    long_opt = False

    # init mixture
    if verbose: print('choosing first kernel')
    tmp_ksd = np.zeros(N)
    for n in range(N):
        tmp_ksd[n] = ksd(logp = logp, y = y[n,:].reshape(1, K), T = np.array([t_increment]), w = np.ones(1), up = up, kernel_sampler = kernel_sampler, B = B)
        # end for

    argmin = np.argmin(tmp_ksd) # ksd minimizer
    if verbose: print('first sample point chosen: ' + str(y[argmin]))
    w[argmin] = 1 # update weight
    T[argmin] = t_increment # update steps
    active = np.array([argmin]) # update active locations, kernel_sampler
    #if verbose: print('number of steps: ' + str(T))
    obj = np.array([ksd(logp = logp, y = y[argmin,:].reshape(1, K), T = np.array([t_increment]), w = np.ones(1), up = up, kernel_sampler = kernel_sampler, B = 10000)]) # update objective
    if verbose: print('ksd: ' + str(obj[-1]))


    # plot initial approximation
    if plot:
        if verbose: print('plotting')
        plotting(y, T, w, logp, plot_path, iter_no = 0, kernel_sampler = kernel_sampler, plt_lims = plt_lims, N = 10000)

    active_kernels = np.array([1.])
    cpu_time = np.array([time.perf_counter() - t0])

    if verbose:
        print('cpu time: ' + str(cpu_time[-1]))
        print()


    for iter_no in range(maxiter):

        if verbose:
            print('iteration ' + str(iter_no + 1))
            print('assessing convergence')
        if convergence: break


        if verbose: print('choosing next kernel')
        argmin = choose_kernel(up, logp, y, active, T, t_increment, t_max, w, B = B, kernel_sampler = kernel_sampler)
        if verbose: print('chosen sample point: ' + str(y[argmin]))

        # update steps
        T[argmin] = T[argmin] + t_increment
        #if verbose: print('number of steps: ' + str(T))


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
            w[active] = weight_opt(logp, y, T, w, active, up, kernel_sampler = kernel_sampler, t_increment = t_increment, tol = 0, b = w_schedule(iter_no), B = B, maxiter = w_maxiters(iter_no, long_opt), verbose = verbose, trace = trace, tracepath = plot_path + 'weight_trace/')
        else:
            if verbose: print('not updating weights')


        # estimate objective
        if verbose: print('estimating objective function')
        obj = np.append(obj, ksd(logp = logp, y = y, T = T, w = w, up = stop_up, kernel_sampler = kernel_sampler, B = 10000))

        # update convergence
        if verbose: print('updating convergence')
        if np.abs(obj[-1]) < tol: convergence = True

        # plot current approximation
        if plot:
            if verbose: print('plotting')
            plotting(y, T, w, logp, plot_path, iter_no = iter_no + 1, kernel_sampler = kernel_sampler, plt_lims = plt_lims, N = 10000)

        # calculate cumulative computing time and active kernels
        cpu_time = np.append(cpu_time, time.perf_counter() - t0)
        active_kernels = np.append(active_kernels, active.shape[0])

        if verbose:
            print('number of active kernels: ' + str(active_kernels[-1]))
            print('active sample: ' + str(np.squeeze(y[active,:])))
            print('active steps: ' + str(T[active]))
            print('active weights: ' + str(w[active]))
            print('ksd: ' + str(obj[-1]))
            print('cumulative cpu time: ' + str(cpu_time[-1]))
            print()

        # end for

    if verbose:
        print('done!')
        print('number of active kernels: ' + str(active_kernels[-1]))
        print('sample: ' + str(np.squeeze(y)))
        print('weights: ' + str(w))
        print('steps: ' + str(T))
        print('ksd: ' + str(obj[-1]))

    return w, T, obj, cpu_time, active_kernels
