# suite of functions for doing locally-adapted boosting variational inference

# preamble
import numpy as np
import scipy.stats as stats
import pandas as pd
import time, bisect
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import cProfile, pstats, io
import os
import imageio


###################################
def mix_sample(size, y, T, w, logp, kernel_sampler, t_increment, chains = None, verbose = False):
    """
    sample from mixture of mcmc kernels

    inputs:
        - size number of samples
        - y kernel locations
        - T array with number of steps per kernel location
        - w array with location weights
        - logp target logdensity
        - kernel_sampler is a function that generates samples from the mixture kernels
        - t_increment is an integer with the incremental steps per iteration
        - chains is a list of N shape(T_n,K) arrays with current chains
        - verbose is boolean, indicating whether to print messages during sampling (used for debugging)
    outputs:
        - shape(size,K) array with the sample
    """

    #if verbose:
    #print('mixture sample')
    #print('size: ' + str(size))
    #print('y: ' + str(y))
    #print('w: ' + str(w))
    #print('T: ' + str(T))
    #print('chains: ' + str(chains))
    #print()

    active = w>0
    tmp_y = y[active,:]
    tmp_w = w[active]
    tmp_T = T[active]
    #print('arange active: ' + str(np.arange(y.shape[0])[active]))
    if chains is not None:
        tmp_chains = [chains[n] for n in np.arange(y.shape[0])[active]]
    else:
        tmp_chains = chains

    #print('tmp_y: ' + str(tmp_y))
    #print('tmp_w: ' + str(tmp_w))
    #print('tmp_T: ' + str(tmp_T))
    #print('tmp_chains: ' + str(tmp_chains))
    N = tmp_y.shape[0]
    K = tmp_y.shape[1]

    #with np.random.choice = slow
    #inds = np.random.choice(N, size = size, p = w, replace = True) # indices to sample from
    #values, counts = np.unique(inds, return_counts = True) # sampled values with counts
    #print('original values: ' + str(values))
    #print('original counts: ' + str(counts))

    values = np.arange(N)
    counts = np.floor(size*tmp_w).astype(int)
    counts[-1] += size - counts.sum()
    #print('values: ' + str(values))
    #print('counts: ' + str(counts))

    out = np.zeros((1,K))
    #for i in range(values.shape[0]):
    for idx in values:
        # for each value, generate a sample of size counts[i]
        # now get the active chain to be used in sampling
        #print('idx : ' + str(idx))
        if chains is not None:
            active_chain = [tmp_chains[idx]]
        else:
            active_chain = tmp_chains

        #print('y: ' + str(tmp_y[idx, :]))
        #print('T: ' + str(tmp_T[idx]))
        #print('chain: ' + str(active_chain))
        #print()
        tmp_out = kernel_sampler(y = np.array(tmp_y[idx, :]).reshape(1, K), T = np.array([tmp_T[idx]]), S = counts[idx], logp = logp, t_increment = t_increment, chains = active_chain).reshape(counts[idx], K)

        # add to sample
        out = np.concatenate((out, tmp_out))
    # end for
    return out[1:,:]
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
        - off-diagonal mean of Up, which is (N,N)
    """

    def anon_up(x, y, verbose = False):
        # x, y shape(N,K)
        # out is scalar, with mean of off-diagonal elements of up(x,y) (which itself is (N,N))

        # get all matrices
        N = x.shape[0]
        kr = kernel(x,y)
        dkx = dk_x(x,y)
        dky = dk_y(x,y)
        dkxy = dk_xy(x,y)
        sx = sp(x)
        sy = sp(y)

        term1 = kr*np.squeeze(np.matmul(sx[:,np.newaxis,np.newaxis,:], sy[np.newaxis,:,:,np.newaxis]))
        term2 = np.squeeze(np.matmul(sx[:,np.newaxis,np.newaxis,:], dky[:,:,:,np.newaxis]))
        term3 = np.squeeze(np.matmul(sy[:,np.newaxis,np.newaxis,:], dkx[:,:,:,np.newaxis]))
        tmp = term1 + term2 + term3 + dkxy

        #return(np.sum(tmp) - np.trace(tmp))/(N*(N-1)) # u-statistic; unbiased but can be negative
        return np.mean(tmp) # v-statistic; biased but always positive

    return anon_up
###################################


###################################
def ksd(logp, y, T, w, up, kernel_sampler, t_increment, chains = None, B = 1000):
    """
    estimate ksd

    inputs:
        - logp target logdensity
        - y kernel locations
        - T array with number of steps per kernel location
        - w array with location weights
        - up function to calculate expected value of
        - kernel_sampler is a function that generates samples from the mixture kernels
        - t_increment is an integer with the incremental number of steps per iteration
        - chains is a list of N shape(T_n,K) arrays with current chains
        - B number of MC samples
    outputs:
        - scalar, the estimated ksd
    """

    # generate samples
    X = mix_sample(B, y = y, T = T, w = w, logp = logp, kernel_sampler = kernel_sampler, t_increment = t_increment, chains = chains) # sample from mixture
    X = np.where(~np.isnan(X), X, 0)
    X = np.where(~np.isinf(X), X, 0)
    return up(X, X)
###################################



###################################
def kl(logp, p_sample, y, T, w, up, kernel_sampler, t_increment, chains = None, B = 1000, direction = 'forward'):
    """
    estimate kl

    inputs:
        - logp target logdensity
        - p_sample generates samples from the target density
        - y kernel locations
        - T array with number of steps per kernel location
        - w array with location weights
        - up function to calculate expected value of
        - kernel_sampler is a function that generates samples from the mixture kernels
        - t_increment is an integer with the incremental number of steps per iteration
        - chains is a list of N shape(T_n,K) arrays with current chains
        - B number of MC samples
        - direction is a string with the kl direction - one of 'forward' or 'reverse'
    outputs:
        - scalar, the estimated kl
    """

    if direction == 'forward':
        if p_sample is None: return np.inf
        ps = p_sample(B)
        qs = mix_sample(10000, y = y, T = T, w = w, logp = logp, kernel_sampler = kernel_sampler, t_increment = t_increment, chains = chains) # sample from mixture
        q = stats.gaussian_kde(np.squeeze(qs), bw_method = 0.25)
        #return max(0.,np.mean(logp(ps) - np.log(q.evaluate(np.squeeze(ps)))))
        return np.mean(logp(ps) - q.logpdf(np.squeeze(ps)))
    elif direction == 'reverse':
        qs = mix_sample(B, y = y, T = T, w = w, logp = logp, kernel_sampler = kernel_sampler, t_increment = t_increment, chains = chains) # sample from mixture
        q = stats.gaussian_kde(qs.T, bw_method = 0.25)
        #return max(0.,np.mean(np.log(q.evaluate(np.squeeze(qs))) - logp(qs)))
        return np.mean(q.logpdf(qs.T) - logp(qs))
###################################



###################################
def simplex_project(x):
    """
    project shape(N,) array x into the probabilistic simplex Delta(N-1)
    returns a shape (N,) array y

    code adapted from Duchi et al. (2008)
    """

    initx = np.copy(x)
    N = x.shape[0]
    mu = -np.sort(-x) # sort x in descending order
    rho_aux = mu - (1 / np.arange(1, N+1)) * (np.cumsum(mu) - 1) # build array to get rho
    rho = bisect.bisect_left(-rho_aux, 0) # first element greater than 0
    theta = (np.sum(mu[:rho]) - 1) / rho #
    x = np.maximum(x - theta, 0)
    if np.any(np.isinf(x)):
        # if overflow, do norm1 projection of absolute value with logsumexp-ish trick
        x = initx
        x = np.abs(x)
        x[x == 0] = 1e-50 # stability
        xmax = np.max(x)
        x = (x/xmax)/np.exp(np.log(x) - np.log(xmax)).sum() # divide by max

    return x
###################################


###################################
def plotting(y, T, w, logp, plot_path, iter_no, t_increment, kernel_sampler = None, plt_lims = None, N = 10000):
    """
    function that plots the target density and approximation, used in each iteration of the main routine

    inputs:
        - y array with kernel locations
        - T array with step sizes
        - w array with weights
        - logp target log density
        - plot_path string with plath to save figures in
        - iter_no integer with iteration number for title
        - t_increment is an integer with the incremental number of steps per iteration
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
            lbvi_sample = np.squeeze(mix_sample(N, y, T, w, logp, kernel_sampler = kernel_sampler, t_increment = t_increment, verbose = False))
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
        cp = plt.contour(xx, yy, np.exp(lp), colors = 'black', levels = 4)
        hcp,_ = cp.legend_elements()
        hcps = [hcp[0]]
        legends = ['p(x)']

        if kernel_sampler is None:
            plt.scatter(y[:,0], y[:,1], marker='.', c='k', alpha = 0.2, label = '')
        else:
            # generate and plot approximation
            lbvi_sample = mix_sample(N, y, T, w, logp, kernel_sampler = kernel_sampler, t_increment = t_increment)
            lbvi_sample = lbvi_sample[~np.isnan(lbvi_sample).any(axis=-1)] # remove nans
            lbvi_sample = lbvi_sample[~np.isinf(lbvi_sample).any(axis=-1)] # remove infs
            if lbvi_sample.size != 0:
                 # otherwise no plottting to do =(
                 lbvi_kde = stats.gaussian_kde(lbvi_sample.T, bw_method = 0.05).evaluate(tt.T).reshape(nn, nn).T
                 cp_lbvi = plt.contour(xx, yy, lbvi_kde, levels = 8, colors = '#39558CFF')
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
def gif_plot(plot_path):
    """
    function that takes all plots generated by plotting script and merges them into a gif
    code from https://stackoverflow.com/questions/41228209/making-gif-from-images-using-imageio-in-python
    """
    #jpg_dir = inpath #+ 'plots/'
    jpg_dir = os.listdir(plot_path)
    jpg_dir = np.setdiff1d(jpg_dir, 'weight_trace') # get rid of weight trace directory
    jpg_dir = np.setdiff1d(jpg_dir, 'tmp') # get rid of weight trace directory
    number = np.zeros(len(jpg_dir))
    i = 0
    # fix names so they are in correct order
    for x in jpg_dir:
        x = x[5:]
        x = x[:-4]
        number[i] = int(x)
        i = i+1
    # end for


    db = pd.DataFrame({'file_name': jpg_dir,
                       'number': number})
    db = db.sort_values(by=['number'])

    images = []
    for file_name in db.file_name:
        images.append(imageio.imread(plot_path + file_name))
    imageio.mimsave(plot_path + 'evolution.gif', images, fps = 2)

###################################

###################################
def w_grad(up, logp, y, T, w, B, kernel_sampler, t_increment, chains = None, X = None):
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
        - t_increment is an integer with the incremental steps per iteration
        - chains is a list of N shape(T_n,K) arrays with current chains
        - X is either None or a shape(B,N,K) array with samples to use for gradient calculation
            if None, samples will be generated; else, provided samples will be used
    outputs:
        - shape(y.shape[0],) array, the gradient
    """
    N = y.shape[0]
    K = y.shape[1]
    grad_w = np.zeros(N)
    if X is None: X = kernel_sampler(y = y, T = T, S = 2*B, logp = logp, t_increment = t_increment, chains = chains)

    # retrieve samples from each kernel and add to gradient
    for n in range(N):
        g = 0
        for i in range(N): g += w[i] * (up(X[:,n,:],X[:,i,:]) + up(X[:,i,:],X[:,n,:]))
        grad_w[n] = g
    # end for

    return grad_w
###################################


###################################
def weight_opt(logp, y, T, w, active, up, kernel_sampler, t_increment, chains = None, tol = 0.001, b = 0.1, B = 1000, maxiter = 1000, sample_recycling = False, verbose = False, trace = False, tracepath = ''):
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
        - t_inrement is an integer with the incremental number of steps per lbvi iteration
        - chains is a list of N shape(T_n,K) arrays with current chains
        - tol is the tolerance for convergence assessment
        - b is the optimization schedule
        - B number of MC samples
        - maxiter bounds the number of algo iterations
        - sample_recycling is boolean indicating whether to generate a single sample and recycle or to generate samples at each iteration
        - verbose is boolean indicating whether to print messages
        - trace is boolean indicating whether to print a trace plot of the objective function
        - tracepath is the path in which the trace plot is saved if generated
    outputs:
        - array with optimal weights of shape active.shape[0]
    """
    # if only one location is active, weight is 1
    if np.unique(active).shape[0] == 1: return np.array([1])

    # subset active locations and chains
    if chains is not None:
        active_chains = [chains[n] for n in active]
    else:
        active_chains = None

    y = y[active,:]
    T = T[active]
    w = w[active]
    w[w == 0] = 0.1
    #w = np.ones(w.shape[0]) / w.shape[0]
    w = w / w.sum()
    n = active.shape[0]

    # create matrix K for gradient
    X = kernel_sampler(y = y, T = T, S = B, logp = logp, t_increment = t_increment, chains = active_chains)
    K = np.zeros((n,n))
    for i in range(n):
        for j in range(n): K[i,j] = up(X[:,i,:],X[:,j,:])
        # end for
    # end for
    K += K.T

    # init algo
    convergence = False
    w_step = 0
    obj = np.array([])

    # run optimization
    for k in range(maxiter):
        if verbose: print('weight opt iter: ' + str(k+1) + '/' + str(maxiter), end = '\r') # to visualize number of iterations
        if convergence: break
        Dw = K@w
        w_step = - (b/np.sqrt(k+1)) * Dw # step size without momentum
        w += w_step # update weight
        w = simplex_project(w) # project to simplex
        #if verbose:
        #    print('Dw: ' + str(Dw))
        #    print('step: ' + str(w_step))
        #    print('w: ' + str(w))
        #    print()
        if np.linalg.norm(Dw) < tol: convergence = True # update convergence

        #if trace:
        #    obj = np.append(obj, ksd(logp = logp, y = y, T = T, w = w, up = up, kernel_sampler = kernel_sampler, t_increment = t_increment, chains = chains, B = B))
        #    #if verbose: print('ksd: ' + str(obj[-1]))
        #    plt.clf()
        #    plt.plot(1 + np.arange(obj.shape[0]), obj, '-k')
        #    plt.xlabel('iteration')
        #    plt.ylabel('kernelized stein discrepancy')
        #    plt.title('trace plot of ksd in weight optimization')
        #    plt.savefig(tracepath + str(np.sum(T) / t_increment) + '.jpg', dpi = 300)
    # end for
    if verbose: print('weights optimized in ' + str(k+1) + ' iterations')
    return w
###################################


###################################
def choose_kernel(up, logp, y, active, T, t_increment, t_max, chains, w, B, kernel_sampler, b, verbose = False):
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
        - chains is a list of N shape(T_n,K) arrays with current chains
        - w array with location weights
        - B number of MC samples
        - kernel_sampler is a function that generates samples from the mixture kernels
        - b is the step size for one step of sgd
    outputs:
        - integer with location that minimizes linear approximation
    """

    N = y.shape[0]
    new_objs = np.zeros(N)

    for n in range(N):
        if verbose: print(str(n+1) + '/' + str(N), end = '\r')

        # define new mixture by increasing steps of chain n
        tmp_w = np.copy(w)
        tmp_T = np.copy(T)
        tmp_T[n] += t_increment

        # limit number of steps
        if tmp_T[n] > t_max:
            new_objs[n] = np.inf
            continue

        # increase weight if necessary
        if tmp_w[n] == 0:
            tmp_w[n] = 0.1/(1+tmp_T[n]/t_increment)
            tmp_w = tmp_w / tmp_w.sum()

        # update active chains
        tmp_active = np.copy(active)
        if n not in active: tmp_active = np.append(tmp_active,n)
        tmp_active = np.sort(tmp_active)
        tmp_chains = [chains[i] for i in tmp_active] if chains is not None else None

        # do 100 steps of weight optimization
        #tmp_w  = weight_opt(logp, y, tmp_T, tmp_w, tmp_active, up, kernel_sampler = kernel_sampler, t_increment = t_increment, chains = chains, tol = 0, b = b, B = B, maxiter = 100, sample_recycling = False, verbose = False, trace = False)

        tmp_w = tmp_w[tmp_active]

        # calculate decrement
        new_objs[n] = ksd(logp, y[tmp_active,:], tmp_T[tmp_active], tmp_w, up, kernel_sampler, t_increment, tmp_chains, B = B)
    # end for

    #print('sample: ' + str(np.squeeze(y)))
    #print('ksds: ' + str(new_objs))
    return np.argmin(new_objs)
###################################



###################################
def lbvi(y, logp, t_increment, t_max, up, kernel_sampler, w_maxiters = None, w_schedule = None, B = 1000, maxiter = 100, tol = 0.001, stop_up = None, weight_max = 20, cacheing = True, result_cacheing = False, sample_recycling = False, verbose = False, plot = True, plt_lims = None, plot_path = 'plots/', trace = False, gif = True, p_sample = None):
    """
    locally-adaptive boosting variational inference main routine
    given a sample and a target, find the mixture of user-defined kernels that best approximates the target

    inputs:
        - y                : shape(N,K) array of kernel locations (sample)
        - logp             : function, the target log density
        - t_increment      : integer with number of steps to increase chain by
        - t_max            : integer with max number of steps allowed per chaikernel_samplern
        - up               : function to calculate expected value of when estimating ksd
        - kernel_sampler   : function that generates samples from the mixture kernels
        - w_maxiters       : function that receives the iteration number and a boolean indicating whether opt is long or not and outputs the max number of iterations for weight optimization
        - w_schedule       : function that receives the iteration number and outputs the step size for the weight optimization
        - B                : number of MC samples for estimating ksd and gradients
        - maxiter          : integer with the max the number of algo iterations
        - tol              : float with the tolerance below which the algorithm breaks the loop and stops
        - stop_up          : indicates the stopping criterion. If None, the ksd provided will be used. Else, provide an auxiliary up function, which will be used to build a surrogate ksd to determine convergence
        - weight_max       : integer indicating max number of iterations without weight optimization (if no new kernels are added to mixture)
        - cacheing         : boolean indicating whether mc samples should be cached
        - result_cacheing  : boolean indicating whether intermediate results should be stored (recommended for large jobs)
        - sample_recycling : boolean indicating whether to generate a single sample and recycle or to generate samples at each iteration
        - verbose          : boolean indicating whether to print messages
        - plot             : boolean indicating whether to generate plots of the approximation at each iteration (only supported for uni and bivariate data)
        - plt_lims         : array with the plotting limits (xinf, xsup, yinf, ysup)
        - trace            : boolean indicating whether to print a trace plot of the objective function
        - plot_path        : the path in which the trace plot is saved if generated
        - gif              : boolean indicating whether a gif with the plots will be created (only if plot is also True)
        - p_sample         : function that generates samples from the target distribution (for calculating reverse kl in synthetic experiments) or None to be ignored

    outputs:
        - w, T : shape(y.shape[0], ) arrays with the sample, the weights, and the steps sizes
        - obj  : array with the value of the objective function at each iteration
    """
    if verbose:
        print('running locally-adaptive boosting variational inference')
        print()
        #print('alphas: ' + str(np.squeeze(y[:,0])))
        #print()

    t0 = time.perf_counter()
    N = y.shape[0]
    K = y.shape[1]
    kls = np.inf

    # init array with chain arrays
    if cacheing:
        chains = [y[n,:].reshape(1,K) for n in range(N)]
    else:
        chains = None
    # chains[n] is a shape(T_n,K) np array with the chain of kernel n at time T - used for cacheing

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
    if plot: plotting(y, T = 0, w = 0, logp = logp, plot_path = plot_path, iter_no = -1, t_increment = t_increment, kernel_sampler = None, plt_lims = plt_lims, N = 10000)

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
        if verbose: print(str(n+1) + '/' + str(N), end = '\r')
        tmp_ksd[n] = ksd(logp = logp, y = y[n,:].reshape(1, K), T = np.array([t_increment]), w = np.ones(1), up = up, kernel_sampler = kernel_sampler, t_increment = t_increment, chains = None, B = 1000)
        # end for

    argmin = np.argmin(tmp_ksd) # ksd minimizer
    if verbose: print('first sample point chosen: ' + str(y[argmin, 0:min(K,3)]))
    w[argmin] = 1 # update weight
    T[argmin] = t_increment # update steps
    active = np.array([argmin]) # update active locations, kernel_sampler
    #if verbose: print('number of steps: ' + str(T))

    # now update chains with the new increlogp(ps) - np.log(q.evaluate(np.squeeze(ps)))ment
    if cacheing:
        if verbose: print('updating chains')
        _, chains = kernel_sampler(y, T, 1, logp, t_increment = t_increment, chains = chains, update = True)
        #if verbose: print('current chains: ' + str(chains))


    # estimate objective function
    if verbose: print('estimating objective function')
    obj_timer0 = time.perf_counter() # to not time obj estimation
    obj = np.array([ksd(logp = logp, y = y[argmin,:].reshape(1, K), T = np.array([t_increment]), w = np.ones(1), up = up, kernel_sampler = kernel_sampler, t_increment = t_increment, chains = chains, B = 1000)]) # update objective
    if verbose: print('ksd: ' + str(obj[-1]))
    if p_sample is not None:
        kls = np.array([kl(logp, p_sample, y, T, w, up, kernel_sampler, t_increment, chains = None, B = 1000, direction = 'reverse')])
        if verbose: print('KL: ' + str(kls[-1]))
    obj_timer = time.perf_counter() - obj_timer0


    # plot initial approximation
    if plot:
        if verbose: print('plotting')
        plotting(y, T, w, logp, plot_path, iter_no = 0, t_increment = t_increment, kernel_sampler = kernel_sampler, plt_lims = plt_lims, N = 10000)

    active_kernels = np.array([1.])
    cpu_time = np.array([time.perf_counter() - t0 - obj_timer])

    # save results
    if result_cacheing:
        if not os.path.exists(plot_path + 'tmp/'): os.makedirs(plot_path + 'tmp/')
        np.save(plot_path + 'tmp/tmp_T.npy', T)
        np.save(plot_path + 'tmp/tmp_w.npy', w)
        np.save(plot_path + 'tmp/tmp_obj.npy', obj)
        np.save(plot_path + 'tmp/tmp_kernels.npy', active_kernels)
        np.save(plot_path + 'tmp/tmp_cput.npy', cpu_time)

    if verbose:
        print('cpu time: ' + str(cpu_time[-1]))
        print()


    for iter_no in range(maxiter):

        if verbose:
            print('iteration ' + str(iter_no + 1))
            print('assessing convergence')
        if convergence: break


        if verbose: print('choosing next kernel')
        argmin = choose_kernel(up, logp, y, active, T, t_increment, t_max, chains = chains, w = w, B = B, kernel_sampler = kernel_sampler, b = w_schedule(1), verbose = verbose)
        if verbose: print('chosen sample point: ' + str(y[argmin, 0:min(K,3)]))

        # update steps
        T[argmin] = T[argmin] + t_increment
        #if verbose: print('number of steps: ' + str(T))

        # update chains
        if cacheing:
            if verbose: print('updating chains')
            _, chains = kernel_sampler(y, T, 1, logp, t_increment, chains = chains, update = True)
            #if verbose: print('new chains: ' + str(chains))


        # update active set and determine weight length
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

        active = np.sort(active) # for sampling purposes further down the road

        # update weights
        if update_weights:
            if verbose: print('updating weights')
            w[active] = weight_opt(logp, y, T, w, active, up, kernel_sampler = kernel_sampler, t_increment = t_increment, chains = chains, tol = 0, b = w_schedule(iter_no), B = B, maxiter = w_maxiters(iter_no, long_opt), sample_recycling = sample_recycling, verbose = verbose, trace = trace, tracepath = plot_path + 'weight_trace/')
        else:
            if verbose: print('not updating weights')


        # estimate objective
        if verbose: print('estimating objective function')
        obj_timer0 = time.perf_counter() # to not time obj estimation
        obj = np.append(obj, ksd(logp = logp, y = y, T = T, w = w, up = stop_up, kernel_sampler = kernel_sampler, t_increment = t_increment, chains = None, B = 1000))
        if p_sample is not None:
            if verbose: print('estimating kl')
            kls = np.append(kls, kl(logp, p_sample, y, T, w, up, kernel_sampler, t_increment, chains = None, B = 1000, direction = 'reverse'))
        obj_timer = time.perf_counter() - obj_timer0
        if verbose: print('objective function estimated in ' + str(obj_timer) + ' seconds')

        # update convergence
        if verbose: print('updating convergence')
        if np.abs(obj[-1]) < tol: convergence = True

        # plot current approximation
        if plot:
            if verbose: print('plotting')
            try:
                plotting(y, T, w, logp, plot_path, iter_no = iter_no + 1, t_increment = t_increment, kernel_sampler = kernel_sampler, plt_lims = plt_lims, N = 10000)
            except:
                if verbose: print('plotting failed')

        # calculate cumulative computing time and active kernels
        cpu_time = np.append(cpu_time, time.perf_counter() - obj_timer - t0)
        active_kernels = np.append(active_kernels, w[w>0].shape[0])

        # save results
        if result_cacheing:
            np.save(plot_path + 'tmp/tmp_T.npy', T)
            np.save(plot_path + 'tmp/tmp_w.npy', w)
            np.save(plot_path + 'tmp/tmp_obj.npy', obj)
            np.save(plot_path + 'tmp/tmp_kernels.npy', active_kernels)
            np.save(plot_path + 'tmp/tmp_cput.npy', cpu_time)


        if verbose:
            print('number of active kernels: ' + str(active_kernels[-1]))
            print('active sample: ' + str(np.squeeze(y[active, 0:min(K,3)])))
            print('active steps: ' + str(T[active]))
            print('active weights: ' + str(w[active]))
            print('ksd: ' + str(obj[-1]))
            if p_sample is not None: print('KL: ' + str(kls[-1]))
            print('cumulative cpu time: ' + str(cpu_time[-1]))
            #print('current chains: ' + str(chains))
            print()

        # end for

    if plot and gif:
        if verbose: print('plotting gif')
        gif_plot(plot_path)


    if verbose:
        print('done!')
        print('number of active kernels: ' + str(active_kernels[-1]))
        print('sample: ' + str(np.squeeze(y)))
        print('weights: ' + str(w))
        print('steps: ' + str(T))
        print('ksd: ' + str(obj[-1]))
        if p_sample is not None: print('KL: ' + str(kls[-1]))
        print('cumulative cpu time: ' + str(cpu_time[-1]))

    return w, T, obj, cpu_time, active_kernels, kls



###################################
def choose_kernel_old(up, logp, y, active, T, t_increment, t_max, chains, w, B, kernel_sampler, verbose = False):
    """
    DEPRECATED; SEE NEW VERSION
    choose kernel to add to the mixture

    inputs:
        - up function to calculate expected value of
        - logp target logdensity
        - y kernel locations
        - active is an array with active locations
        - T array with number of steps per kernel location
        - t_increment integer with number of steps to increase chain by
        - t_max max number of steps allowed per chain
        - chains is a list of N shape(T_n,K) arrays with current chains
        - w array with location weights
        - B number of MC samples
        - kernel_sampler is a function that generates samples from the mixture kernels
    outputs:
        - integer with location that minimizes linear approximation
    """


    N = y.shape[0]
    grads = np.zeros(N)

    for n in range(N):
        if verbose: print(str(n+1) + '/' + str(N), end = '\r')

        # settings:
        #tmp_active = np.setdiff1d(active, np.array([n])) # if chain is active, remove. else, do nothing
        tmp_active = np.copy(active)

        # calculate exactly if this is the only active chain
        if tmp_active.size == 0:
            d0 = ksd(logp = logp, y = np.array([y[n]]), T = np.array([T[n]]), w = np.ones(1), up = up, kernel_sampler = kernel_sampler, t_increment = t_increment, chains = chains, B = B)
            d1 = ksd(logp = logp, y = np.array([y[n]]), T = np.array([T[n] + t_increment]), w = np.ones(1), up = up, kernel_sampler = kernel_sampler, t_increment = t_increment, chains = chains, B = B)
            grads[n] = d0 - d1 # exact decrease
            break
        #print('active chains: ' + str(tmp_active))
        #print('active locations: ' + str(y[tmp_active]))

        tmp_w = np.copy(w)
        #tmp_w[n] = 0                   # the chain to be run is removed from mixture
        tmp_w = tmp_w[tmp_active]       # only active weights
        tmp_w = simplex_project(tmp_w)  # and weights normalized
        #print('active weights: ' + str(tmp_w))

        tmp_T = np.copy(T)
        tmp_T[n] = tmp_T[n] + t_increment # increase number of steps in kernel n

        if tmp_T[n] > t_max:
            grads[n] = np.inf
        else:
            #tmp_T = tmp_T[tmp_active]
            #print('active steps: ' + str(tmp_T[tmp_active]))
            # generate samples
            X_mix = mix_sample(size = 4*B, y = y[active,:], T = T, w = w[active], logp = logp, kernel_sampler = kernel_sampler, t_increment = t_increment)
            X_kernel = kernel_sampler(y = np.array([y[n,:]]), T = np.array([tmp_T[n]]), S = 2*B, logp = logp, t_increment = t_increment)[:,0,:]

            Dw = w_grad(up, logp, y, T, w, B, kernel_sampler = kernel_sampler, t_increment = t_increment, chains = active_chains, mix_X = mix_X, X = X) # get gradient
            #print(Dw)
            #w_step = 0.9*w_step - (b/np.sqrt(k+1)) * Dw # step size with momentum
            w_step = - (b/np.sqrt(k+1)) * Dw # step size without momentum
            w += w_step # update weight
            w = simplex_project(w) # project to simplex

            # estimate gradient
            grads[n] = up(X_mix[:B,:], X_kernel[:B,:]).mean() + up(X_kernel[B:2*B,:], X_mix[B:2*B,:]).mean() - 2*up(X_mix[2*B:3*B,:], X_mix[3*B:4*B,:]).mean()

    # end for
    #print('sample: ' + str(np.squeeze(y)))
    #print('gradients: ' + str(grads))
    return np.argmin(grads)
###################################


###################################
def w_grad_old(up, logp, y, T, w, B, kernel_sampler, t_increment, chains = None, mix_X = None, X = None):
    """
    DEPRECATED; SEE NEW VERSION
    calculate gradient of the KSD wrt to the weights

    inputs:
        - up function to calculate expected value of
        - logp target logdensity
        - y shape(N,K) kernel locations
        - T array with number of steps per kernel location
        - w array with location weights
        - B number of MC samples
        - kernel_sampler is a function that generates samples from the mixture kernels
        - t_increment is an integer with the incremental steps per iteration
        - chains is a list of N shape(T_n,K) arrays with current chains
        - mix_X, X are either None or shape(2*B,K) arrays with samples to use for gradient calculation
            if None, samples will be generated; else, provided samples will be used
    outputs:
        - shape(y.shape[0],) array, the gradient
    """

    N = y.shape[0]
    K = y.shape[1]

    # init
    grad_w = np.zeros(N)

    if mix_X is None:
        # sample from the mixture
        #print('sampling from mixture')
        mix_X = mix_sample(2*B, y = y, T = T, w = w, logp = logp, kernel_sampler = kernel_sampler, t_increment = t_increment, chains = chains)
    mix_X = mix_X[~np.isnan(mix_X).any(axis=-1)] # remove nans
    mix_X = mix_X[~np.isinf(mix_X).any(axis=-1)] # remove infs

    if X is None:
        # sample from kernels
        #print('w grad y: ' + str(y))
        X = kernel_sampler(y = y, T = T, S = 2*B, logp = logp, t_increment = t_increment, chains = chains)
    X = np.where(~np.isnan(X), X, 0) # remove nans
    X = np.where(~np.isinf(X), X, 0) # remove infs

    # get new size
    if mix_X.shape[0] % 2 == 1:
        B_new1 = int((mix_X.shape[0]-1)/2)
    else:
        B_new1 = int(mix_X.shape[0]/2)
    if X.shape[0] % 2 == 1:
        B_new2 = int((mix_X.shape[0]-1)/2)
    else:
        B_new2 = int(mix_X.shape[0]/2)
    B = np.minimum(B_new1, B_new2)
    mix_Y = mix_X[-B:]
    mix_X = mix_X[:B]


    # sample from each kernel and define gradient
    for n in range(y.shape[0]):
        tmp_X = X[:,n,:]
        tmp_Y = tmp_X[-B:,:]
        tmp_X = tmp_X[:B,:]

        # get gradient
        grad_w[n] = up(tmp_X, mix_X) + up(mix_Y, tmp_Y)
    # end for

    return grad_w
###################################


###################################
def weight_opt_old(logp, y, T, w, active, up, kernel_sampler, t_increment, chains = None, tol = 0.001, b = 0.1, B = 1000, maxiter = 1000, sample_recycling = False, verbose = False, trace = False, tracepath = ''):
    """
    DEPRECATED; SEE NEW VERSION
    optimize weights via sgd

    inputs:
        - logp target density
        - y shape(N,K) kernel locations
        - T array with number of steps per kernel location
        - w array with location weights
        - active array with number of active locations
        - up function to calculate expected value of
        - kernel_sampler is a function that generates samples from the mixture kernels
        - t_inrement is an integer with the incremental number of steps per lbvi iteration
        - chains is a list of N shape(T_n,K) arrays with current chains
        - tol is the tolerance for convergence assessment
        - b is the optimization schedule
        - B number of MC samples
        - maxiter bounds the number of algo iterations
        - sample_recycling is boolean indicating whether to generate a single sample and recycle or to generate samples at each iteration
        - verbose is boolean indicating whether to print messages
        - trace is boolean indicating whether to print a trace plot of the objective function
        - tracepath is the path in which the trace plot is saved if generated
    outputs:
        - array with optimal weights of shape active.shape[0]
    """

    # if only one location is active, weight is 1
    if np.unique(active).shape[0] == 1: return np.array([1])


    # subset active locations and chains
    if chains is not None:
        active_chains = [chains[n] for n in active]
    else:
        active_chains = None

    y = y[active,:]
    T = T[active]
    w = w[active]
    w[w == 0] = 0.1
    #w = np.ones(w.shape[0]) / w.shape[0]
    w = w / w.sum()
    n = active.shape[0]

    #print('active chains: ' + str(active_chains))
    #print('active sample: ' + str(y))
    #print('active steps: ' + str(T))
    #print('active weights: ' + str(w))

    # generate recyclable samples
    if sample_recycling:
        if verbose: print('generating samples for recycling')
        X = kernel_sampler(y = y, T = T, S = B, logp = logp, t_increment = t_increment, chains = active_chains)
    else:
        X = None

    # init algo
    convergence = False
    w_step = 0
    obj = np.array([])

    # run optimization
    for k in range(maxiter):
        #print(w)
        if verbose: print('weight opt iter: ' + str(k+1) + '/' + str(maxiter), end = '\r') # to visualize number of iterations
        if convergence: break # assess convergence
        Dw = w_grad(up, logp, y, T, w, B, kernel_sampler = kernel_sampler, t_increment = t_increment, chains = active_chains, X = X) # get gradient
        #print(Dw)
        #w_step = 0.9*w_step - (b/np.sqrt(k+1)) * Dw # step size with momentum
        w_step = - (b/np.sqrt(k+1)) * Dw # step size without momentum
        w += w_step # update weight
        w = simplex_project(w) # project to simplex
        if verbose:
            print('Dw: ' + str(Dw))
            print('step: ' + str(w_step))
            print('w: ' + str(w))
            print()
        if np.linalg.norm(Dw) < tol: convergence = True # update convergence

        #if trace:
        #    obj = np.append(obj, ksd(logp = logp, y = y, T = T, w = w, up = up, kernel_sampler = kernel_sampler, t_increment = t_increment, chains = chains, B = B))
        #    #if verbose: print('ksd: ' + str(obj[-1]))
        #    plt.clf()
        #    plt.plot(1 + np.arange(obj.shape[0]), obj, '-k')
        #    plt.xlabel('iteration')
        #    plt.ylabel('kernelized stein discrepancy')
        #    plt.title('trace plot of ksd in weight optimization')
        #    plt.savefig(tracepath + str(np.sum(T) / t_increment) + '.jpg', dpi = 300)


    # end for


    if verbose:
        print('weights optimized in ' + str(k+1) + ' iterations')

    return w
###################################
