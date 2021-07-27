# suite of functions for doing locally-adapted boosting variational inference with SMC components

# preamble
import numpy as np
import scipy.stats as stats
import pandas as pd
import time, bisect
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import os
import imageio

##########################
##########################
## auxiliary functions ###
##########################
##########################
def kl(logq, logp, sampler, B = 1000, direction = 'reverse'):
    """
    Estimate the KL divergence
    Input:
    logq       : function, log density of variational approximation
    logp       : function, log target density
    sampler    : function, generates samples from either q or p (depending on direction)
    B          : int, number of samples to generate
    direction  : str, either reverse or forward

    Output:
    kl         : float, estimate of KL(q||p) if direction is reverse, and of KL(p||q) if direction is forward
    """
    theta = sampler(B)
    if direction == 'reverse':
        return np.mean(logq(theta) - logp(theta), axis=-1)
    elif direction == 'forward':
        return np.mean(logp(theta) - logq(theta), axis=-1)
    else: raise NotImplementedError


def kl_mixture(y, w, samples, beta, Zs, logp, direction = 'reverse'):
    """
    Estimate the KL divergence
    Input:
    y          : (N,K) array with sample locations
    w          : (N,) array with component weights
    samples    : list of arrays, samples from each component
    beta       : (N,) array with betas of each component
    Zs         : (N,) array with normalizing constants of each component
    logp       : function, log target density
    B          : int, number of samples to generate
    direction  : str, either reverse or forward

    Output:
    kl         : float, estimate of KL(q||p) if direction is reverse, and of KL(p||q) if direction is forward
    """
    if direction == 'reverse':
        logq = lambda x : mix_logpdf(x, logp, y, w, None, 1., beta, None, 1., Zs)
        obj = 0.
        for n in range(y.shape[0]):
            if w[n] == 0: continue # don't waste time evaluating the logpdfs if they are not going to add
            print(w[n])
            print(logq(samples[n]))
            print(logp(samples[n]))
            obj += w[n]*np.mean(logq(samples[n]) - logp(samples[n]), axis=-1)
        return obj
    else: raise NotImplementedError

def logsumexp(x):
    # logsumexp over last axis of x
    maxx = np.amax(x,axis=-1)
    return maxx + np.log(np.sum(np.exp(x-maxx[...,np.newaxis]),axis=-1))



##########################
##########################
## plotting functions  ###
##########################
##########################
def plotting(logp, y, w, smc, r_sd, beta, beta_ls, plt_name, plt_lims, B = 10000, Zs = None):
    """
    Plot the current approximation against the target (for K <= 2)
    Input:
    logp      : function, target log density (for SMC)
    y         : (N,K) array, component locations
    w         : (N,) array, contains weights of each component
    smc       : function, generates samples via SMC
    r_sd      : float, std deviation of reference distributions
    beta      : (N,) array, contains betas of each component
    beta_ls   : list of arrays, each array contains the discretization of each component
    plt_name  : str, path, file name, and extension in single string to save plot
    plt_lims  : (4,) array, plot limits (also used to generate linspaces). Should be [x_inf, x_sup, y_inf, y_sup]
    B         : int, number of particles in SMC
    Zs        : (N,) array or None, contains normalizing constants or None (in which case smc is run to estimate normalizing constants)
    """
    plt.clf()

    # univariate data plotting
    if y.shape[1] == 1:
        # get plotting limits
        x_lower = plt_lims[0]
        x_upper = plt_lims[1]
        y_upper = plt_lims[3]

        # plot target density
        t = np.linspace(x_lower, x_upper, 1000)
        p = np.exp(logp(t[:,np.newaxis]))
        plt.plot(t, p, linestyle = 'solid', color = 'black', label = 'p(x)', lw = 3)


        # generate and plot approximation
        q = np.exp(mix_logpdf(t[:,np.newaxis], logp, y, w, smc, r_sd, beta, beta_ls, B, Zs))
        plt.plot(t, q, linestyle = 'dashed', color = '#39558CFF', label='q(x)', lw = 3)

        # beautify and save plot
        plt.ylim(0, y_upper)
        plt.xlim(x_lower, x_upper)
        plt.legend(fontsize = 'medium', frameon = False)
        plt.savefig(plt_name, dpi = 300)

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


        # generate and plot approximation
        lq = mix_logpdf(tt, logp, y, w, smc, r_sd, beta, beta_ls, B, Zs).reshape(nn, nn).T
        q = np.exp(lq)
        cq = plt.contour(xx, yy, np.exp(lq), colors = '#39558CFF', levels = 4)
        hcq,_ = cq.legend_elements()
        hcps.append(hcq[0])
        legends.append('q(x)')

        # beautify and save plot
        plt.ylim(y_lower, y_upper)
        plt.xlim(x_lower, x_upper)
        plt.legend(hcps, legends, fontsize = 'medium', frameon = False)
        plt.savefig(plt_name, dpi = 300)


def gif_plot(plot_path):
    """
    Function that takes all plots generated by plotting script and merges them into a gif
    Code from https://stackoverflow.com/questions/41228209/making-gif-from-images-using-imageio-in-python
    Input:
    plot_path : str, path in which plots are stored; gif will be saved in same directory
    """
    #jpg_dir = inpath #+ 'plots/'
    jpg_dir = os.listdir(plot_path)
    jpg_dir = np.setdiff1d(jpg_dir, 'weight_trace') # get rid of weight trace directory
    jpg_dir = np.setdiff1d(jpg_dir, 'tmp')          # get rid of tmp files directory
    number = np.zeros(len(jpg_dir))
    i = 0
    # fix names so they are in correct order
    for x in jpg_dir:
        x = x[5:]
        x = x[:-4]
        number[i] = int(x)
        i = i+1
    # end for

    db = pd.DataFrame({'file_name': jpg_dir, 'number': number})
    db = db.sort_values(by=['number'])

    images = []
    for file_name in db.file_name:
        images.append(imageio.imread(plot_path + file_name))
    imageio.mimsave(plot_path + 'lbvi.gif', images, fps = 5)


##########################
##########################
## Gaussian functions ####
##########################
##########################
def norm_logpdf(x, mean, sd):
    """
    log PDF of isotropic Normal(loc, sd x I_K)
    Input:
    x    : (N,K) array, pdf will be evaluated at x; last axis corresponds to dimension and has shape K
    mean : (K,) array, mean of the distribution
    sd   : float, isotropic standard deviation
    """
    K = x.shape[-1]
    return -0.5*np.sum((x-mean)**2,axis=-1)/sd**2 -0.5*K*np.log(2*np.pi) - 0.5*K*np.log(sd)

def norm_random(B, mean, sd):
    """
    Generate samples from isotropic Normal(loc, sd x I_K)
    Input:
    B    : int, number of samples to draw
    mean : shape(K,) array, mean of the distribution
    sd   : float, isotropic standard deviation
    """
    K = mean.shape[0]
    return sd*np.random.randn(B,K) + mean


##########################
##########################
### mixture functions  ###
##########################
##########################
def smc_logqn(x, logr, logp, beta, Z):
    """
    Evaluate log pdf of nth component
    Input:
    x     : nd-array, point at which to evaluate logpdf; last axis corresponds to dimension
    logr  : function, reference log density
    logp  : function, target log density
    beta  : float, discretization temperature
    Z     : float, normalizing constant

    Out:
    lp    : (N,) array, log probabilities at x
    """
    return (1-beta)*logr(x)+beta*logp(x)-np.log(Z)



def mix_sample(size, logp, y, w, smc, r_sd, beta, beta_ls):
    """
    Sample from mixture of SMC components
    Input:
    size    : int, size of the sample
    logp    : function, target log density (for SMC)
    y       : (N,K) array, component locations
    w       : (N,) array, contains weights of each component
    smc     : function, generates samples via SMC
    r_sd    : float, std deviation of reference distributions
    beta    : (N,) array, contains betas of each component
    beta_ls : list of arrays, each array contains the discretization of each component

    Output:
    sample  : (size,K) array with sample from mixture
    """

    active = w>0
    tmp_y = y[active,:]
    tmp_w = w[active]
    tmp_beta = beta[active]

    N = tmp_y.shape[0]
    K = tmp_y.shape[1]

    values = np.arange(N)
    counts = np.floor(size*tmp_w).astype(int)
    counts[-1] += size - counts.sum()

    if np.any(counts < 0):
        print('Error, negative counts. Counts: ' + str(counts))
        print('size: ' + str(size))
        print('weights: ' + str(tmp_w))

    out = np.zeros((1,K))
    for idx in values:
        # for each value, generate a sample of size counts[idx]
        tmp_logr = lambda x : norm_logpdf(x, tmp_y[idx,:], r_sd)
        tmp_r_sample = lambda B : norm_random(B, tmp_y[idx,:], r_sd)
        tmp_beta_ls = beta_ls[idx]
        tmp_beta_ls = tmp_beta_ls[tmp_beta_ls <= tmp_beta[idx]]
        tmp_out,_,_ = smc(logp = logp, logr = tmp_logr, r_sample = tmp_r_sample, B = counts[idx], beta_ls = tmp_beta_ls, Z0 = 1)
        out = np.concatenate((out, tmp_out)) # add to sample
    # end for
    return out[1:,:]

def mix_logpdf(x, logp, y, w, smc, r_sd, beta, beta_ls, B, Z):
    """
    Evaluate log pdf of mixture of SMC components
    Input:
    x       : nd-array, point at which to evaluate logpdf; last axis corresponds to dimension
    logp    : function, target log density (for SMC)
    y       : (N,K) array, component locations
    w       : (N,) array, contains weights of each component
    smc     : function, generates samples via SMC
    r_sd    : float, std deviation of reference distributions
    beta    : (N,) array, contains betas of each component
    beta_ls : list of arrays, each array contains the discretization of each component
    B       : int, number of particles to use in SMC
    Z       : (N,) array or None, contains normalizing constants or None (in which case smc is run to estimate normalizing constants)

    Output:
    sample  : (size,K) array with sample from mixture
    """
    # filter out weights == 0
    active = w>0
    tmp_y = y[active,:]
    tmp_w = w[active]
    tmp_beta = beta[active]
    Z = Z[active]

    N = tmp_y.shape[0]
    K = tmp_y.shape[1]
    lps = np.zeros((x.shape[0],N))

    for n in range(N):
        tmp_logr = lambda x : norm_logpdf(x, tmp_y[n,:], r_sd)
        if Z is None:
            # for each value, estimate normalizing constant
            tmp_r_sample = lambda B : norm_random(B, tmp_y[n,:], r_sd)
            tmp_beta_ls = beta_ls[n]
            tmp_beta_ls = tmp_beta_ls[tmp_beta_ls <= tmp_beta[n]]

            # run smc and use Z estimate to evaluate logpdf
            _,tmp_Z,_ = smc(logp = logp, logr = tmp_logr, r_sample = tmp_r_sample, B = B, beta_ls = tmp_beta_ls, Z0 = 1)
        else:
            # just use provided estimate
            tmp_Z = Z[n]

        # evaluate logpdf of nth component
        lps[:,n] = smc_logqn(x, tmp_logr, logp, tmp_beta[n], tmp_Z) + np.log(tmp_w[n])
    # end for
    return logsumexp(lps)


##########################
##########################
####  beta functions  ####
##########################
##########################

## gradient calculation ###
def kl_grad_beta(b, logp, y, w, beta, beta_ls, r_sd, smc, B, n):
    """
    First derivative of KL wrt beta for component n evaluated at b
    Input: see choose_beta

    Output:
    float, stochastic estimate of KL gradient
    """

    beta[n] = b
    beta_ls = [beta_ls[i][beta_ls[i] <= beta[i]] for i in range(y.shape[0])]

    # generate sample from nth component
    logr = lambda x : norm_logpdf(x, mean = y[n,:], sd = r_sd)
    r_sample = lambda B : norm_random(B, mean = y[n,:], sd = r_sd)
    tmp_beta_ls = beta_ls[n]
    theta,_,_ = smc(logp = logp, logr = logr, r_sample = r_sample, B = B, beta_ls = tmp_beta_ls, Z0 = 1)
    logq = lambda x : mix_logpdf(x, logp, y, w, smc, r_sd, beta, beta_ls, B)

    lp = logp(theta)
    return w[n]*np.cov(lp-logr(theta), logq(theta)-lp)[0][1]


def kl_grad2_beta(b, logp, y, w, beta, beta_ls, r_sd, smc, B, n):
    """
    First derivative of KL wrt beta for component n evaluated at 0
    Input: see choose_beta

    Output:
    float, stochastic estimate of KL gradient
    """

    beta[n] = b
    beta_ls = [beta_ls[i][beta_ls[i] <= beta[i]] for i in range(y.shape[0])]

    # generate sample from nth component and calculate logpdf
    logr = lambda x : norm_logpdf(x, mean = y[n,:], sd = r_sd)
    r_sample = lambda B : norm_random(B, mean = y[n,:], sd = r_sd)
    tmp_beta_ls = beta_ls[n]
    theta,Z,_ = smc(logp = logp, logr = logr, r_sample = r_sample, B = B, beta_ls = tmp_beta_ls, Z0 = 1)
    logqn = lambda x : ((1-beta[n])*logr(x) + beta[n]*logp(x)) - np.log(Z)
    logq = lambda x : mix_logpdf(x, logp, y, w, smc, r_sd, beta, beta_ls, B)

    logrbyp = logr(theta) - logp(theta)
    logqbyp = logq(theta) - logp(theta)
    g = w[n]*np.exp(logqn(theta))/np.exp(logq(theta)) + logqbyp
    g = g*(logrbyp - np.mean(logrbyp))
    term1 = w[n]*np.cov(logrbyp, g)[0][1]
    term2 = w[n]*np.mean(logqbyp)*np.var(logrbyp)

    return term1-term2


## choosing next component based on weight ###
def choose_beta(logp, y, w, beta, beta_ls, r_sd, smc, b_gamma, B, verbose = False):
    """
    Choose component that results in greatest KL decrease due to beta perturbation
    Input:
    logp    : function, log target density
    y       : (N,K) array with locations
    w       : (N,) array, weights of components
    beta    : (N,) array, betas of components
    beta_ls : list of np arrays, contains discretizations of components
    r_sd    : float, std deviation of reference distributions
    smc     : function, generates samples via SMC
    b_gamma : float, newton's step size
    B       : integer, number of particles ot use in SMC and to estimate gradients
    verbose : boolean, whether to print messages

    Output:
    argmin  : component that minimizes the KL
    disc    : estimate of the KL at the optimal component and alpha
    """

    N = y.shape[0]
    K = y.shape[1]
    trimmed_beta_ls = [beta_ls[n][beta_ls[n] <= beta[n]] for n in range(N)]
    smc_eps = beta_ls[0][0]
    kls = np.zeros(N)
    beta_star = np.zeros(N)

    for n in range(N):
        if verbose: print(str(n+1) + '/' + str(N), end='\r')
        if w[n] == 0:
            # can't perturb beta if component is not active
            kls[n] = np.inf
        else:
            # calculate minimizer of second order approximation to kl
            grad = kl_grad_beta(beta[n], logp, y, w, beta, trimmed_beta_ls, r_sd, smc, B, n)
            grad2 = kl_grad2_beta(beta[n], logp, y, w, beta, trimmed_beta_ls, r_sd, smc, B, n)
            beta_star[n] = beta[n]-b_gamma*grad/grad2
            beta_star[n] = min(1, max(smc_eps, beta_star[n])) # iteratively scale until feasible point

            # calculate kl estimate at minimizer
            tmp_trimmed_beta_ls = trimmed_beta_ls.copy()
            tmp_trimmed_beta_ls[n] = beta_ls[n][beta_ls[n] <= beta_star[n]] # trim nth discretization up to beta_star instead of beta[n]
            tmp_logq = lambda x : mix_logpdf(x, logp, y, w, smc, r_sd, beta, tmp_trimmed_beta_ls, B)
            tmp_sampler = lambda B : mix_sample(B, logp, y, w, smc, r_sd, beta, tmp_trimmed_beta_ls)
            kls[n] = kl(logq = tmp_logq, logp = logp, sampler = tmp_sampler, B = B)

            print('y: ' + str(y[n,:]) + '   |   β: ' + str(beta_star[n]) + '   |   Gradient: ' + str(grad) + '   |   Hessian: ' + str(grad2) + '   |   KL: ' + str(kls[n]))
        # end if
    # end for
    argmin = np.argmin(kls)
    return argmin, kls[argmin], beta_star[argmin]





##########################
##########################
#### weight functions ####
##########################
##########################

## gradient calculation ###
def kl_grad_alpha(alpha, logp, y, w, beta, beta_ls, r_sd, smc, B, samples, Zs, n):
    """
    First derivative of KL wrt alpha for component n evaluated at alpha
    Input: see choose_weight

    Output:
    float, stochastic estimate of KL gradient
    """
    if w[n] == 1: return 0.

    beta_ls = [beta_ls[i][beta_ls[i] <= beta[i]] for i in range(y.shape[0])]

    # generate sample from nth component and define logpdf
    tmp_logr = lambda x : norm_logpdf(x, mean = y[n,:], sd = r_sd)
    if Zs is None:
        tmp_r_sample = lambda B : norm_random(B, mean = y[n,:], sd = r_sd)
        tmp_beta_ls = beta_ls[n]
        theta1,Z1,_ = smc(logp = logp, logr = tmp_logr, r_sample = tmp_r_sample, B = B, beta_ls = tmp_beta_ls, Z0 = 1)
    else:
        theta1 = samples[n]
        Z1 = Zs[n]
    logqn = lambda x : smc_logqn(x, tmp_logr, logp, beta[n], Z1)


    # generate sample from mixture minus nth component and define logpdf
    tmp_w = np.copy(w)
    tmp_w[n] = 0.
    tmp_w = tmp_w / (1-w[n]) # normalize
    logq = lambda x : mix_logpdf(x, logp, y, tmp_w, smc, r_sd, beta, beta_ls, B, Zs)

    # define gamma
    def gamma_n(theta):
        exponents = np.column_stack((np.log((1-alpha)*w[n]+alpha) + logqn(theta), np.log(1-alpha) + np.log(1-w[n]) + logq(theta)))
        return logsumexp(exponents) - logp(theta)

    if Zs is None:
        theta2 = mix_sample(B, logp, y, tmp_w, smc, r_sd, beta, beta_ls)
        return (1-w[n])*np.mean(gamma_n(theta1) - gamma_n(theta2))
    else:
        out = gamma_n(theta1)
        for k in range(y.shape[0]):
            if tmp_w[k] == 0: continue
            out -= tmp_w[n]*gamma_n(samples[n])
        return (1-w[n])*np.mean(out)


def kl_grad2_alpha(logp, y, w, beta, beta_ls, r_sd, smc, B, samples, Zs, n):
    """
    Second derivative of KL wrt alpha at component n, evaluated at 0
    Input: see choose_weight

    Output:
    float, stochastic estimate of KL second derivative
    """
    if w[n] == 1: return 0.

    beta_ls = [beta_ls[i][beta_ls[i] <= beta[i]] for i in range(y.shape[0])]
    logq_full = lambda x : mix_logpdf(x, logp, y, w, smc, r_sd, beta, beta_ls, B, Zs)

    # generate sample from nth component and define logpdf
    tmp_logr = lambda x : norm_logpdf(x, mean = y[n,:], sd = r_sd)
    tmp_r_sample = lambda B : norm_random(B, mean = y[n,:], sd = r_sd)
    tmp_beta_ls = beta_ls[n]
    theta1,Z1,_ = smc(logp = logp, logr = tmp_logr, r_sample = tmp_r_sample, B = B, beta_ls = tmp_beta_ls, Z0 = 1)
    logqn = lambda x : ((1-beta[n])*tmp_logr(x) + beta[n]*logp(x)) - np.log(Z1)


    # generate sample from mixture minus nth component and define logpdf
    tmp_w = np.copy(w)
    tmp_w[n] = 0.
    tmp_w = tmp_w / (1-w[n]) # normalize
    logq = lambda x : mix_logpdf(x, logp, y, tmp_w, smc, r_sd, beta, beta_ls, B)
    theta2 = mix_sample(B, logp, y, tmp_w, smc, r_sd, beta, beta_ls)

    # define psi
    def psi_n(theta): return (np.exp(logqn(theta)) - np.exp(logq(theta))) / np.exp(logq_full(theta))

    return (1-w[n])**2 * np.mean(psi_n(theta1) - psi_n(theta2))



## choosing next component based on weight ###
def choose_weight(logp, y, w, beta, beta_ls, r_sd, smc, w_gamma, B, samples, Zs, verbose = False):
    """
    Choose component that results in greatest KL decrease due to weight perturbation
    Input:
    logp    : function, log target density
    y       : (N,K) array with locations
    w       : (N,) array, weights of components
    beta    : (N,) array, betas of components
    beta_ls : list of np arrays, contains discretizations of components
    r_sd    : float, std deviation of reference distributions
    smc     : function, generates samples via SMC
    w_gamma : float, newton's step size
    B       : integer, number of particles ot use in SMC and to estimate gradients
    samples : list of arrays or None, samples from each component. If None, samples will be generated
    Zs      : (N,) array or None, normalizing constants of each component. If None, they will be calculated
    verbose : boolean, whether to print messages

    Output:
    argmin  : component that minimizes the KL
    disc    : estimate of the KL at the optimal component and alpha
    """

    N = y.shape[0]
    K = y.shape[1]
    beta_ls = [beta_ls[n][beta_ls[n] <= beta[n]] for n in range(N)]
    kls = np.zeros(N)
    grads = np.zeros(N)
    alpha_star = np.zeros(N)

    for n in range(N):
        #if verbose: print(str(n+1) + '/' + str(N), end='\r')
        if w[n] == 1:
            # can't perturb weight if it's the only element in mixture
            logq = lambda x : mix_logpdf(x, logp, y, w, smc, r_sd, beta, beta_ls, B, Zs)
            if Zs is None:
                sampler = lambda B : mix_sample(B, logp, y, w, smc, r_sd, beta, beta_ls)
                kls[n] = kl(logq, logp, sampler, B = B)
            else:
                kls[n] = kl_mixture(y, w, samples, beta, Zs, logp)
        else:
            # calcualte minimizer of second order approximation to kl
            grad = kl_grad_alpha(0., logp, y, w, beta, beta_ls, r_sd, smc, B, samples, Zs, n)
            grads[n] = grad
            grad2 = kl_grad2_alpha(logp, y, w, beta, beta_ls, r_sd, smc, B, samples, Zs, n)
            alpha_star[n] = -w_gamma*grad/grad2
            alpha_star[n] = min(1., max(-w[n]/(1-w[n]), alpha_star[n])) # alpha_star in appropriate range

            # calculate kl estimate at minimizer
            tmp_w = w*(1-alpha_star[n]/(1-w[n]))
            tmp_w[n] = w[n] + alpha_star[n] # optimal value using original w instead of scaled one
            tmp_logq = lambda x : mix_logpdf(x, logp, y, tmp_w, smc, r_sd, beta, beta_ls, B, Zs)
            tmp_sampler = lambda B : mix_sample(B, logp, y, tmp_w, smc, r_sd, beta, beta_ls)
            kls[n] = kl(logq = tmp_logq, logp = logp, sampler = tmp_sampler, B = B)

            print('y: ' + str(y[n,:]) + '   |   α: ' + str(alpha_star[n]) + '   |   Gradient: ' + str(grad) + '   |   Hessian: ' + str(grad2) + '   |   KL: ' + str(kls[n]))
        # end if
    # end for
    argmin = np.argmin(kls)
    return argmin, kls[argmin], alpha_star[argmin]

def weight_opt(alpha_s, n, logp, y, w, beta, beta_ls, r_sd, smc, w_schedule, B = 10000, maxiter = 1000, verbose = False):
    """
    Optimize the weight of nth component via SGD
    Input:
    alpha_s : float, initialization point
    n       : int, component to optimize
    logp    : function, log target density
    y       : (N,K) array with locations
    w       : (N,) array, weights of components
    beta    : (N,) array, betas of components
    beta_ls : list of np arrays, contains discretizations of components
    r_sd    : float, std deviation of reference distributions
    smc     : function, generates samples via SMC
    w_gamma : float, newton's step size
    B       : int, number of particles ot use in SMC and to estimate gradients
    maxiter : int, maximum number of iterations
    verbose : boolean, whether to print messages

    Out:
    w_opt   : (N,) array with optimal weights
    alpha   : float, optimal value of alpha
    """
    alpha = alpha_s
    if verbose:
        tmp_logq = lambda x : mix_logpdf(x, logp, y, w, smc, r_sd, beta, beta_ls, B)
        tmp_sampler = lambda B : mix_sample(B, logp, y, w, smc, r_sd, beta, beta_ls)
        obj = kl(logq = tmp_logq, logp = logp, sampler = tmp_sampler, B = B)
        print('0/' + str(maxiter) + '   |   α: '  + str(alpha) + '   |   Gradient : ' + str(kl_grad_alpha(alpha, logp, y, w, beta, beta_ls, r_sd, smc, B, n)) + '  |   KL: ' + str(obj), end='\n')
    for k in range(maxiter):
        #if verbose: print(str(k+1) + '/' + str(maxiter), end='\r')
        Dalpha = kl_grad_alpha(alpha, logp, y, w, beta, beta_ls, r_sd, smc, B, n)
        alpha -= w_schedule(k+1)*Dalpha
        alpha = min(1.,max(0.,alpha))
        if verbose and (k+1)%50==0:
            # estimate kl and print info
            tmp_w = w*(1-alpha)
            tmp_w[n] = w[n] + alpha
            tmp_logq = lambda x : mix_logpdf(x, logp, y, tmp_w, smc, r_sd, beta, beta_ls, B)
            tmp_sampler = lambda B : mix_sample(B, logp, y, tmp_w, smc, r_sd, beta, beta_ls)
            obj = kl(logq = tmp_logq, logp = logp, sampler = tmp_sampler, B = B)
            print(str(k+1) + '/' + str(maxiter) + '   |   α: '  + str(alpha) + '   |   Gradient : ' + str(Dalpha) + '  |   KL: ' + str(obj), end='\n')
    # end for
    w_opt = (1-alpha)*w
    w_opt[n] += alpha
    return w_opt, alpha



##########################
##########################
#### MAIN FUNCTION #######
##########################
##########################
def lbvi_smc(y, logp, smc, smc_eps = 0.05, r_sd = None, maxiter = 10, w_gamma = 1., w_schedule = None, w_maxiter = 1000, b_gamma = 1., B = 1000, cacheing = False, verbose = False, plot = False, plot_path = '', plot_lims = [-10,10,-10,10], gif = False):
    """
    Run LBVI with SMC components
    Input:
    y          : (N,K) array, initial sample locations
    logp       : function, target log density
    smc        : function, smc sampler (see readme in tests/smc/)
    smc_eps    : float, step size for smc discretization
    r_sd       : float, std deviation used for reference distributions; if None, 3 will be used
    maxiter    : int, maximum number of iterations to run the main loop for
    w_gamma    : float, newton's step size for weight optimization
    w_schedule : function or None, receives iteration number and returns step size (defaults to 1/sqrt(iter) if None)
    w_maxiter  : int, maximum number of iterations for step size optimization
    b_gamma    : float, newton's step size for beta optimization
    B          : int, number of MC samples to use for gradient estimation
    verbose    : boolean, whether to cache samples and logpdfs; speeds up algorithm but might decrease accuracy
    verbose    : boolean, whether to print messages
    plot       : boolean, whether to plot approximation vs target at each iteration
    plot_path  : str, folder in which plots should be saved (ignored if plot == False)
    plot_lims  : (4,) array, plot limits; see plotting function documentation
    gif        : boolean, whether to create a gif with the approximation plots; only done if plot=True as well

    Output:
    """
    if verbose:
        print('Running LBVI with SMC components')
        print()

    # init timer and initialize values
    t0 = time.perf_counter()
    N = y.shape[0]
    K = y.shape[1]
    beta_ls = [np.linspace(0,1,int(1/smc_eps)+1) for n in range(N)]
    betas = np.zeros(N)
    w = np.zeros(N)
    if w_schedule is None: w_schedule = lambda k : 1./np.sqrt(k)
    samples = [norm_random(B, y[n,:], r_sd) for n in range(N)] if cacheing else None  # init at reference dist
    Zs = np.ones(N)  # all normalizing constants are 1


    ##########################
    ##########################
    # initialize mixture #####
    ##########################
    ##########################
    if verbose: print('Optimizing first component')
    if r_sd is None: r_sd = 3

    tmp_kl = np.zeros(N)
    for n in range(N):
        if verbose: print(str(n+1) + '/' + str(N), end = '\r')
        tmp_sampler = lambda B : samples[n] if cacheing else norm_random(B, y[n,:], r_sd)
        tmp_logq = lambda x : norm_logpdf(x, y[n,:], r_sd)
        tmp_kl[n] = kl(logq = tmp_logq, logp = logp, sampler = tmp_sampler, B = B)
    # end for

    argmin = np.argmin(tmp_kl)    # kl minimizer
    if verbose: print('First component mean: ' + str(y[argmin, 0:min(K,3)]))
    w[argmin] = 1.                # update weight
    active = np.array([argmin])   # init active set


    ##########################
    ##########################
    # estimate objective #####
    ##########################
    ##########################
    obj_timer = time.perf_counter()
    tmp_sampler = lambda B : norm_random(B, y[argmin,:], r_sd)
    tmp_logq = lambda x : norm_logpdf(x, y[argmin,:], r_sd)
    obj = np.array([kl(logq = tmp_logq, logp = logp, sampler = tmp_sampler, B = 100000)])
    obj_timer = time.perf_counter() - obj_timer


    ##########################
    ##########################
    # plot approximation #####
    ##########################
    ##########################
    plt_timer = time.perf_counter()
    if plot:
        if verbose: print('Plotting approximation')
        plt_name = plot_path + 'iter_0.jpg'
        plotting(logp, y, w, smc, r_sd, betas, beta_ls, plt_name, plot_lims, B = 10000, Zs = Zs)
    plt_timer = time.perf_counter() - plt_timer

    ##########################
    ##########################
    # stats print out    #####
    ##########################
    ##########################
    cpu_time = np.array([time.perf_counter() - t0 - obj_timer - plt_timer])
    active_kernels = np.array([1.])
    if verbose:
        print('KL: ' + str(obj[-1]))
        print('CPU time: ' + str(cpu_time[-1]))
        print()


    ##########################
    ##########################
    # start main loop    #####
    ##########################
    ##########################
    for iter in range(1,maxiter+1):
        if verbose: print('Iteration ' + str(iter) + '/' + str(maxiter))

        # calculate optimal weight perturbation
        if verbose: print('Determining optimal α')
        w_argmin,w_disc,alpha_s = choose_weight(logp, y, w, betas, beta_ls, r_sd, smc, w_gamma, B, samples, Zs, verbose)

        # calculate optimal beta perturbation
        if verbose: print('Determining optimal β')
        beta_argmin,beta_disc,beta_s = choose_beta(logp, y, w, betas, beta_ls, r_sd, smc, b_gamma, B, verbose)

        if verbose: print('Preliminary (α*, β*) = (' + str(alpha_s) + ', ' + str(beta_s) + ')')

        # determine whether to perturb weight or beta and update active set
        if w_disc < beta_disc:
            if verbose: print('Optimizing the weight of ' + str(y[w_argmin]))
            #w = (1-alpha_s)*w                     # evenly scale down weights
            #w[w_argmin] = w[w_argmin] + alpha_s   # add alpha bit to argmin weight
            active = np.append(active, w_argmin)
            w,alpha_s = weight_opt(alpha_s, w_argmin, logp, y, w, betas, beta_ls, r_sd, smc, w_schedule, B, w_maxiter, verbose)
            if verbose: print('Optimal α*: ' + str(alpha_s))
        else:
            if verbose: print('Modifying the beta of ' + str(y[beta_argmin]))
            betas[beta_argmin] = beta_s
            active = np.append(active, beta_argmin)

        # update mixture
        active = np.unique(active)
        logq = lambda x : mix_logpdf(x, logp, y, w, smc, r_sd, betas, beta_ls, B)
        q_sampler = lambda B : mix_sample(B, logp, y, w, smc, r_sd, betas, beta_ls)

        # estimate objective function
        obj_timer = time.perf_counter()
        obj = np.append(obj, kl(logq = logq, logp = logp, sampler = q_sampler, B = 100000))
        obj_timer = time.perf_counter() - obj_timer

        # plot approximation
        plt_timer = time.perf_counter()
        if plot:
            if verbose: print('Plotting approximation')
            plt_name = plot_path + 'iter_' + str(iter) + '.jpg'
            plotting(logp, y, w, smc, r_sd, betas, beta_ls, plt_name, plot_lims, B = 10000)
        plt_timer = time.perf_counter() - plt_timer

        # update cpu times and active components
        cpu_time = np.append(cpu_time, time.perf_counter() - t0 - obj_timer - plt_timer)
        active_kernels = np.append(active_kernels, w[w>0].shape[0])

        # stats printout
        if verbose:
            print('Active components: ' + str(np.squeeze(y[active, 0:min(K,3)])))
            print('Weights: ' + str(w[active]))
            print('Betas: ' + str(betas[active]))
            print('KL: ' + str(obj[-1]))
            print('# of active kernels: ' + str(active_kernels[-1]))
            print('CPU time: ' + str(cpu_time[-1]))
            print('User elapsed time: ' + str(time.perf_counter()-t0))
            print()
    # end for

    # generate gif plot
    if plot and gif:
        if verbose: print('Generating gif')
        gif_plot(plot_path)

    return y, w, betas, obj, cpu_time, active_kernels
