# suite of functions for doing locally-adapted boosting variational inference
# with uniform step size increase

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
from lbvi import mix_sample, ksd, kl, simplex_project, plotting, gif_plot, w_grad, weight_opt




###################################
def choose_kernel(up, logp, y, active, T, t_increment, chains, w, B, kernel_sampler, b, verbose = False):
    """
    choose kernel to add to the mixture

    inputs:
        - up function to calculate expected value of
        - logp target logdensity
        - y kernel locations
        - active is an array with active locations
        - T array with number of steps per kernel location
        - t_increment integer with number of steps to increase chain by
        - chains is a list of N shape(T_n,K) arrays with current chains
        - w array with location weights
        - B number of MC samples
        - kernel_sampler is a function that generates samples from the mixture kernels
        - b is the step size for one step of sgd
    outputs:
        - integer with location that minimizes ksd
    """

    N = y.shape[0]
    inactive = np.setdiff1d(np.arange(N), active) # inactive components
    new_objs = np.zeros(inactive.shape[0])

    ##########################
    ##########################
    ### adding one kernel ####
    ##########################
    ##########################
    for i in range(inactive.shape[0]):
        if verbose: print(str(i+1) + '/' + str(inactive.shape[0]), end = '\r')
        n = inactive[i]

        # define new mixture by increasing steps of chain n
        tmp_w = np.copy(w)
        tmp_T = np.copy(T)
        tmp_T[n] = T[active[0]]

        # increase weight
        tmp_w[n] = 0.1/(1+tmp_T[n]/t_increment)
        tmp_w = tmp_w / tmp_w.sum()

        # update active chains
        tmp_active = np.append(active,n)
        tmp_active = np.sort(tmp_active)
        tmp_chains = [chains[i] for i in tmp_active] if chains is not None else None

        # calculate decrement
        new_objs[i] = ksd(logp, y[tmp_active,:], tmp_T[tmp_active], tmp_w[tmp_active], up, kernel_sampler, t_increment, tmp_chains, B)
    # end for

    #######################
    #######################
    ### increasing all ####
    #######################
    #######################
    tmp_ksd = ksd(logp, y[active,:], T[active]+t_increment, w[active], up, kernel_sampler, t_increment, chains, B)
    if tmp_ksd < np.amin(new_objs):
        return active[0] # if increasing all is better, return the first element of active
    else:
        return inactive[np.argmin(new_objs)] # if not, return the inactive component that minimizes ksd
###################################


###################################
def ulbvi(y, logp, t_increment, up, kernel_sampler, w_maxiters = None, w_schedule = None, B = 1000, maxiter = 100, tol = 0.001, stop_up = None, weight_max = 20, cacheing = True, result_cacheing = False, verbose = False, plot = True, plt_lims = None, plot_path = 'plots/', trace = False, gif = True, p_sample = None):
    """
    locally-adaptive boosting variational inference main routine
    given a sample and a target, find the mixture of user-defined kernels that best approximates the target

    inputs:
        - y                : shape(N,K) array of kernel locations (sample)
        - logp             : function, the target log density
        - t_increment      : integer with number of steps to increase chain by
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
        - verbose          : boolean indicating whether to print messages
        - plot             : boolean indicating whether to generate plots of the approximation at each iteration (only supported for uni and bivariate data)
        - plt_lims         : array with the plotting limits (xinf, xsup, yinf, ysup)
        - trace            : boolean indicating whether to print a trace plot of the objective function
        - plot_path        : the path in which the trace plot is saved if generated
        - gif              : boolean indicating whether a gif with the plots will be created (only if plot is also True)
        - p_sample         : function that generates samples from the target distribution (for calculating reverse kl in synthetic experiments) or None to be ignored

    outputs:
        - w, T           : shape(y.shape[0], ) arrays with the sample, the weights, and the steps sizes
        - obj            : array with the value of the objective function at each iteration
        - cpu_time       : array with the cumulative cpu time at each iteration
        - active_kernels : array with the number of active kernels at each iteration
        - kls            : array with the value of the kl divergence at each iteration
    """


    if verbose:
        print('running uniform locally-adaptive boosting variational inference')
        print()

    # init values
    t0 = time.perf_counter()
    N = y.shape[0]
    K = y.shape[1]
    kls = np.inf

    # init array with chain arrays
    chains = [y[n,:].reshape(1,K) for n in range(N)] if cacheing else None
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

    # now update chains with the new increment
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

    #######################
    #######################
    ##### lbvi loop #######
    #######################
    #######################
    for iter_no in range(maxiter):

        if verbose:
            print('iteration ' + str(iter_no + 1))
            #print('assessing convergence')
        if convergence: break


        if verbose: print('choosing next kernel')
        argmin = choose_kernel(up, logp, y, active, T, t_increment, chains = chains, w = w, B = B, kernel_sampler = kernel_sampler, b = w_schedule(1), verbose = verbose)
        if verbose: print('chosen sample point: ' + str(y[argmin, 0:min(K,3)]))


        # update active set and steps and determine weight opt details
        if argmin not in active:
            T[argmin] = T[active[0]] # if minimizer not active, assign current T
            active = np.append(active, argmin) # update active locations
            update_weights = True
            weight_opt_counter = 0
            long_opt = True
        elif weight_opt_counter > weight_max:
            T[active] += t_increment # if minimizer is active, increase all active ones
            update_weights = True
            weight_opt_counter  = 0
            long_opt = False
        else:
            T[active] += t_increment # if minimizer is active, increase all active ones
            update_weights = False
            weight_opt_counter += 1

        active = np.sort(active) # for sampling purposes further down the road

        # update chains
        if cacheing:
            #if verbose: print('updating chains')
            _, chains = kernel_sampler(y, T, 1, logp, t_increment, chains = chains, update = True)
            #if verbose: print('new chains: ' + str(chains))


        # update weights
        if update_weights:
            if verbose: print('updating weights')
            w[active] = weight_opt(logp, y, T, w, active, up, kernel_sampler = kernel_sampler, t_increment = t_increment, chains = chains, tol = 0, b = w_schedule(iter_no), B = B, maxiter = w_maxiters(iter_no, long_opt), verbose = verbose, trace = trace, tracepath = plot_path + 'weight_trace/')
        else:
            if verbose: print('not updating weights')


        # estimate objective
        #if verbose: print('estimating objective function')
        obj_timer0 = time.perf_counter() # to not time obj estimation
        obj = np.append(obj, ksd(logp = logp, y = y, T = T, w = w, up = stop_up, kernel_sampler = kernel_sampler, t_increment = t_increment, chains = None, B = 1000))
        if p_sample is not None:
            #if verbose: print('estimating kl')
            kls = np.append(kls, kl(logp, p_sample, y, T, w, up, kernel_sampler, t_increment, chains = None, B = 1000, direction = 'reverse'))
        obj_timer = time.perf_counter() - obj_timer0
        #if verbose: print('objective function estimated in ' + str(obj_timer) + ' seconds')

        # update convergence
        #if verbose: print('updating convergence')
        if np.abs(obj[-1]) < tol: convergence = True

        # plot current approximation
        if plot:
            #if verbose: print('plotting')
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
            print('number of active kernels: ' + str(int(active_kernels[-1])))
            print('active sample: ' + str(np.squeeze(y[active, 0:min(K,3)])))
            print('active weights: ' + str(w[active]))
            print('active steps: ' + str(np.unique(T[active])))
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
