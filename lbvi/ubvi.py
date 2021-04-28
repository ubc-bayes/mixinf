# suite of functions for doing universal boosting variational inference
# taken from Trevor Campbell and Xinglong Li (2019) Universal Boosting Variational Inference
# code available (and taken and minimally adapted from) https://github.com/trevorcampbell/ubvi

# log sum exp
#from __future__ import absolute_import
import scipy.misc, scipy.special
from autograd.extend import primitive, defvjp
import autograd.numpy as anp
from autograd.numpy.numpy_vjps import repeat_to_match_shape

import autograd.numpy as np
from scipy.optimize import nnls

# autograd
#import autograd.numpy as np
import time

# BVI
#import autograd.numpy as np
from autograd import grad
#import time
#from ..optimization.adam import adam


# COMMON ####################################
import numpy as cnp
#from ubvi.autograd import logsumexp


def mixture_logpdf(X, mu, Sig, wt):
    if len(Sig.shape) < 3:
        Sig = cnp.array([cnp.diag(Sig[i, :]) for i in range(Sig.shape[0])])
    Siginv = cnp.linalg.inv(Sig)
    inner_prods = (X[:,cnp.newaxis,:]-mu)[:,:,:,cnp.newaxis] * Siginv *  (X[:,cnp.newaxis,:]-mu)[:,:,cnp.newaxis,:]
    lg = -0.5*inner_prods.sum(axis=3).sum(axis=2)
    lg -= 0.5*mu.shape[1]*cnp.log(2*cnp.pi) + 0.5*cnp.linalg.slogdet(Sig)[1]
    return logsumexp(lg[:, wt>0]+cnp.log(wt[wt>0]), axis=1)

def mixture_sample(mu, Sig, wt, n_samples):
    if len(Sig.shape) < 3:
        Sig = cnp.array([cnp.diag(Sig[i, :]) for i in range(Sig.shape[0])])
    cts = cnp.random.multinomial(n_samples, wt)
    X = cnp.zeros((n_samples, mu.shape[1]))
    c = 0
    for k in range(wt.shape[0]):
        X[c:c+cts[k], :] = cnp.random.multivariate_normal(mu[k, :], Sig[k, :, :], cts[k])
        c += cts[k]
    return X

def kl_estimate(mus, Sigs, wts, logp, p_samps, direction='forward'):
    lp = logp(p_samps)
    if direction == 'forward':
        lq = mixture_logpdf(p_samps, mus, Sigs, wts)
        kl = (lp - lq).mean()
    else:
        lq = mixture_logpdf(p_samps, mus, Sigs, wts)
        ratio_max = (lq - lp).max()
        kl = cnp.exp(ratio_max)*((lq - lp)*cnp.exp( (lq-lp) - ratio_max)).mean()
    return kl
####################################


# COMPONENTS ####################################
class Component(object):

    def unflatten(self, params):
        raise NotImplementedError

    def logpdf(self, params, X):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def log_sqrt_pair_integral(self):
        raise NotImplementedError


class Gaussian(Component):

    def __init__(self, d, diag): #d is dimension of the space, diag is whether to use diagonal covariance or full
        self.d = d
        self.diag = diag

    def unflatten(self, params):
        params = np.atleast_2d(params)
        N = params.shape[0]
        mu = params[:,:self.d]
        if self.diag:
            logSig = params[:, self.d:]
            return {"mus": mu, "Sigs": np.exp(logSig)}
        else:
            L = params[:, self.d:].reshape((N, self.d, self.d))
            Sig = np.array([np.dot(l, l.T) for l in L])
            Siginv = np.array([np.linalg.inv(sig) for sig in Sig])
            return {"mus": mu, "Sigs": Sig, "Siginvs": Siginv}

    def logpdf(self, params, X):
        theta = self.unflatten(params)
        if len(X.shape)==1 and self.d==1:
            # need to add a dimension so that each row is an observation
            X = X[:,np.newaxis]
        X = np.atleast_2d(X)
        mu = theta['mus']
        Sig = theta['Sigs']
        if self.diag:
            logp = -0.5*mu.shape[1]*np.log(2*np.pi) - 0.5*np.sum(np.log(Sig), axis=1) - 0.5*np.sum((X[:,np.newaxis,:]-mu)**2/Sig, axis=2)
        else:
            Siginv = theta['Siginvs']
            logp = -0.5*mu.shape[1]*np.log(2*np.pi) - 0.5*np.linalg.slogdet(Sig)[1] - 0.5*((X[:,np.newaxis,:]-mu)*((Siginv*((X[:,np.newaxis,:]-mu)[:,:,np.newaxis,:])).sum(axis=3))).sum(axis=2)
        if logp.shape[1]==1:
            logp = logp[:,0]
        return logp

    def sample(self, param, n):
        std_samples = np.random.randn(n, self.d)
        mu = param[:self.d]
        if self.diag:
            lsig = param[self.d:]
            sd = np.exp(lsig/2)
            return mu + std_samples*sd
        else:
            L = param[self.d:].reshape((self.d, self.d))
            return mu + np.dot(std_samples, L)

    def _get_paired_param(self, param1, param2, flatten=False):
        theta = self.unflatten(np.vstack((param1, param2)))
        mu = theta['mus']
        Sig = theta['Sigs']
        if self.diag:
            Sigp = 2./ (1/Sig[0, :] + 1/Sig[1, :])
            mup = 0.5*Sigp*(mu[0, :]/Sig[0, :] + mu[1, :]/Sig[1, :])
        else:
            Siginv = theta['Siginvs']
            Sigp = 2.0*np.linalg.inv(Siginv[0, :, :]+Siginv[1, :, :])
            mup = 0.5*np.dot(Sigp, np.dot(Siginv[0,:,:], mu[0,:]) + np.dot(Siginv[1,:,:], mu[1,:]))
        if not flatten:
            return mup, Sigp
        else:
            return np.hstack((mup, np.log(Sigp) if self.diag else np.linalg.cholesky(Sigp).flatten()))

    def cross_sample(self, param1, param2, n_samps):
        mup, Sigp = self._get_paired_param(param1, param2)
        if self.diag:
            return mup + np.sqrt(Sigp)*np.random.randn(n_samps, self.d)
        else:
            return np.random.multivariate_normal(mup, Sigp, n_samps)

    def log_sqrt_pair_integral(self, new_param, old_params):
        old_params = np.atleast_2d(old_params)
        mu_new = new_param[:self.d]
        mus_old = old_params[:, :self.d]
        if self.diag:
            lsig_new = new_param[self.d:]
            lsigs_old = old_params[:, self.d:]
            lSig2 = np.log(0.5)+np.logaddexp(lsig_new, lsigs_old)
            return -0.125*np.sum(np.exp(-lSig2)*(mu_new - mus_old)**2, axis=1) - 0.5*np.sum(lSig2, axis=1) + 0.25*np.sum(lsig_new) + 0.25*np.sum(lsigs_old, axis=1)
        else:
            L_new = new_param[self.d:].reshape((self.d, self.d))
            Sig_new = np.dot(L_new, L_new.T)
            N = old_params.shape[0]
            Ls_old = old_params[:, self.d:].reshape((N, self.d, self.d))
            Sigs_old = np.array([np.dot(L, L.T) for L in Ls_old])
            Sig2 = 0.5*(Sig_new + Sigs_old)
            return -0.125*((mu_new - mus_old) * np.linalg.solve(Sig2, mu_new - mus_old)).sum(axis=1) - 0.5*np.linalg.slogdet(Sig2)[1] + 0.25*np.linalg.slogdet(Sig_new)[1] + 0.25*np.linalg.slogdet(Sigs_old)[1]

    def params_init(self, params, weights, inflation):
        params = np.atleast_2d(params)
        i = params.shape[0]
        if i==0:
            mu0 = np.random.multivariate_normal(np.zeros(self.d), inflation*np.eye(self.d))
            if self.diag:
                lSig = np.zeros(self.d)
                xtmp = np.hstack((mu0, lSig))
            else:
                L0 = np.eye(self.d)
                xtmp = np.hstack((mu0, L0.reshape(self.d*self.d)))
        else:
            mu = params[:, :self.d]
            k = np.random.choice(np.arange(i), p=(weights**2)/(weights**2).sum())
            if self.diag:
                lsig = params[:, self.d:]
                mu0 = mu[k,:] + np.random.randn(self.d)*np.sqrt(inflation)*np.exp(lsig[k,:])
                LSig = np.random.randn(self.d) + lsig[k,:]
                xtmp = np.hstack((mu0, LSig))
            else:
                Ls = params[:, self.d:].reshape((i, self.d, self.d))
                sig = np.array([np.dot(L, L.T) for L in Ls])
                mu0 = np.random.multivariate_normal(mu[k,:], inflation*sig[k,:,:])
                L0 = np.exp(np.random.randn())*sig[k,:,:]
                xtmp = np.hstack((mu0, L0.reshape(self.d*self.d)))
        return xtmp

    def print_perf(self, itr, x, obj, grd):
        if itr == 0:
            print("{:^30}|{:^30}|{:^30}|{:^30}|{:^30}".format('Iteration', 'mu', 'Log(Sig)' if self.diag else 'Eigvals(Sig)', 'GradNorm', 'Boosting Obj'))
        if self.diag:
            print("{:^30}|{:^30}|{:^30}|{:^30.3f}|{:^30.3f}".format(itr, str(x[:min(self.d,4)]), str(x[self.d:self.d+min(self.d,4)]), np.sqrt((grd**2).sum()), obj))
        else:
            L = x[self.d:].reshape((self.d,self.d))
            print("{:^30}|{:^30}|{:^30}|{:^30.3f}|{:^30.3f}".format(itr, str(x[:min(self.d,4)]), str(np.linalg.eigvalsh(np.dot(L,L.T))[:min(self.d,4)]), np.sqrt((grd**2).sum()), obj))

####################################



# LOGSUMEXP ####################################
logsumexp = primitive(scipy.special.logsumexp)

def make_grad_logsumexp(ans, x, axis=None, b=1.0, keepdims=False):
    shape, dtype = anp.shape(x), anp.result_type(x)
    def vjp(g):
        g_repeated,   _ = repeat_to_match_shape(g,   shape, dtype, axis, keepdims)
        ans_repeated, _ = repeat_to_match_shape(ans, shape, dtype, axis, keepdims)
        return g_repeated * b * anp.exp(x - ans_repeated)
    return vjp

defvjp(logsumexp, make_grad_logsumexp)
####################################


# AUTOGRAD ####################################
def ubvi_adam(x0, obj, grd, learning_rate, num_iters, callback = None):
    b1=0.9
    b2=0.999
    eps=10**-8
    x = x0.copy()
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    t0 = time.perf_counter()
    for i in range(num_iters):
        g = grd(x, i)
        if callback and (i == 0 or i == num_iters - 1 or (time.perf_counter() - t0 > 0.5)):
            callback(i, x, obj(x, i), g)
            t0 = time.perf_counter()
        m = (1 - b1) * g + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        x = x - learning_rate(i)*mhat/(np.sqrt(vhat) + eps)
    return x
####################################



# BOOSTING VI ####################################
class BoostingVI(object):

    def __init__(self, component_dist, opt_alg, n_init = 10, init_inflation = 100, estimate_error = True, verbose = True):
        self.N = 0 #current num of components
        self.component_dist = component_dist #component distribution object
        self.opt_alg = opt_alg #optimization algorithm function
        self.weights = np.empty(0) #weights
        self.params = np.empty((0, 0))
        self.cput = 0. #total computation time so far
        self.error = np.inf #error for the current mixture
        self.n_init = n_init #number of times to initialize each component
        self.init_inflation = init_inflation #number of times to initialize each component
        self.verbose = verbose
        self.estimate_error = estimate_error

    def build(self, N):
	#build the approximation up to N components
        for i in range(self.N, N):
            t0 = time.perf_counter()
            error_flag = False

            #initialize the next component
            if self.verbose: print("Initializing component " + str(i+1) +"... ")
            x0 = self._initialize()

            #if this is the first component, set the dimension of self.params
            if self.params.size == 0:
                self.params = np.empty((0, x0.shape[0]))
            if self.verbose: print("Initialization of component " + str(i+1)+ " complete, x0 = " + str(x0))

            #build the next component
            if self.verbose: print("Optimizing component " + str(i+1) +"... ")
            grd = grad(self._objective)
            try:
                new_param = self.opt_alg(x0, self._objective, grd)
                if not np.all(np.isfinite(new_param)):
                    raise
                #if np.isnan(new_param).any():
                #    raise
            except: #bbvi can run into bad degeneracies; if so, just revert to initialization and set weight to 0
                if self.verbose: print("Optimization of component failed; reverting to initialization")
                error_flag = True
                new_param = x0
            if self.verbose: print("Optimization of component " + str(i+1) + " complete")

            #add it to the matrix of flattened parameters
            self.params = np.vstack((self.params, new_param))
            if self.verbose: print('Params:' + str(self.component_dist.unflatten(self.params)))

            #compute the new weights and add to the list
            if self.verbose: print('Updating weights...')
            self.weights_prev = self.weights.copy()
            try:
                self.weights = np.atleast_1d(self._compute_weights())
                if not np.all(np.isfinite(self.weights)) or error_flag:
                    raise
            except: #bbvi can run into bad degeneracies; if so, just throw out the new component
                self.weights = np.hstack((self.weights_prev, 0.))

            if self.verbose: print('Weight update complete...')
            if self.verbose: print('Weights: ' + str(self.weights))
            # GC hack to ensure weights in simplex
            if self.weights.shape[0] == 1 and self.weights[0] == 0: self.weights = np.ones(1)

            #compute the time taken for this step
            self.cput += time.perf_counter() - t0

            #estimate current error if desired
            if self.estimate_error:
                err_name, self.error = self._error()

            #print out the current error
            if self.verbose:
                print('Component ' + str(self.params.shape[0]) +':')
                print('Cumulative CPU Time: ' + str(self.cput))
                if self.estimate_error:
                    print(err_name +': ' + str(self.error))
                print('Params:' + str(self.component_dist.unflatten(self.params)))
                print('Weights: ' + str(self.weights))

        #update self.N to the new # comps
        self.N = N

        #generate the nicely-formatted output params
        output = self._get_mixture()
        output['cput'] = self.cput
        output['obj'] = self.error
        return output


    def _initialize(self):
        x0 = None
        obj0 = np.inf
        t0 = time.perf_counter()
        #try initializing n_init times
        for n in range(self.n_init):
            xtmp = self.component_dist.params_init(self.params, self.weights, self.init_inflation)
            objtmp = self._objective(xtmp, -1)
            if objtmp < obj0 or x0 is None:
                x0 = xtmp
                obj0 = objtmp
            if self.verbose and (n == 0 or n == self.n_init - 1 or time.perf_counter() - t0 > 0.5):
                if n == 0:
                    print("{:^30}|{:^30}|{:^30}".format('Iteration', 'Best x0', 'Best obj0'))
                print("{:^30}|{:^30}|{:^30.3f}".format(n, str(x0), obj0))
                t0 = time.perf_counter()
        if x0 is None:
            #if every single initialization had an infinite objective, just raise an error
            raise ValueError
        #return the initialized result
        return x0

    def _compute_weights(self):
        raise NotImplementedError

    def _objective(self, itr):
        raise NotImplementedError

    def _error(self):
        raise NotImplementedError

    def _get_mixture(self):
        raise NotImplementedError
####################################




# UBVI ####################################
class UBVI(BoostingVI):

    def __init__(self, logp, component_dist, opt_alg, n_samples = 100, n_logfg_samples = 100, **kw):
        super().__init__(component_dist, opt_alg, **kw)
        self.logp = logp
        self.n_samples = n_samples
        self.n_logfg_samples = n_logfg_samples
        self.Z = np.empty((0,0))
        self._logfg = np.empty(0)
        self._logfgsum = -np.inf

    def _compute_weights(self):
        #compute new row/col for Z
        Znew = np.exp(self.component_dist.log_sqrt_pair_integral(self.params[-1, :], self.params))

        #expand Z
        Zold = self.Z
        self.Z = np.zeros((self.params.shape[0], self.params.shape[0]))
        self.Z[:-1, :-1] = Zold
        self.Z[-1, :] = Znew
        self.Z[:, -1] = Znew

        #expand logfg
        logfgold = self._logfg
        self._logfg = np.zeros(self.params.shape[0])
        self._logfg[:-1] = logfgold
        self._logfg[-1] = self._logfg_est(self.params[-1, :])

        #compute optimal weights via nnls
        if self.params.shape[0] == 1:
            w = np.array([1])
        else:
            Linv = np.linalg.inv(np.linalg.cholesky(self.Z))
            d = np.exp(self._logfg-self._logfg.max()) #the opt is invariant to d scale, so normalize to have max 1
            b = nnls(Linv, -np.dot(Linv, d))[0]
            lbd = np.dot(Linv, b+d)
            w = np.maximum(0., np.dot(Linv.T, lbd/np.sqrt(((lbd**2).sum()))))

        #compute weighted logfg sum
        #self._logfgsum = logsumexp(np.hstack((-np.inf, self._logfg + np.log(np.maximum(w, 1e-64)))))
        self._logfgsum = logsumexp(np.hstack((-np.inf, self._logfg[w>0] + np.log(w[w>0]))))

        #return the weights
        return w

    def _error(self):
        return "Hellinger Dist Sq", self._hellsq_estimate()

    def _hellsq_estimate(self):
        samples = self._sample_g(self.n_samples)
        lf = 0.5*self.logp(samples)
        lg = self._logg(samples)
        ln = np.log(self.n_samples)
        return 1. - np.exp(logsumexp(lf-lg-ln) - 0.5*logsumexp(2*lf-2*lg-ln))

    def _objective(self, x, itr):
        allow_negative = False if itr < 0 else True

        #lgh = -np.inf if len(self.weights) == 0 else logsumexp(np.log(np.maximum(self.weights[-1], 1e-64)) + self.component_dist.log_sqrt_pair_integral(x, self.params))
        lgh = -np.inf if self.weights.size == 0 else logsumexp(np.log(self.weights[self.weights>0]) + np.atleast_2d(self.component_dist.log_sqrt_pair_integral(x, self.params))[:, self.weights>0])
        h_samples = self.component_dist.sample(x, self.n_samples)
        lf = 0.5*self.logp(h_samples)
        lh = 0.5*self.component_dist.logpdf(x, h_samples)
        ln = np.log(self.n_samples)
        lf_num = logsumexp(lf - lh - ln)
        lg_num = self._logfgsum + lgh
        log_denom = 0.5*np.log(1.-np.exp(2*lgh))
        if lf_num > lg_num:
            logobj = lf_num - log_denom + np.log(1.-np.exp(lg_num-lf_num))
            neglogobj = -logobj
            return neglogobj
        else:
            if not allow_negative:
                return np.inf
            lognegobj = lg_num - log_denom + np.log(1.-np.exp(lf_num-lg_num))
            return lognegobj

    def _logfg_est(self, param):
        samples = self.component_dist.sample(param, self.n_samples)
        lf = 0.5*self.logp(samples)
        lg = 0.5*self.component_dist.logpdf(param, samples)
        ln = np.log(self.n_samples)
        return logsumexp(lf - lg - ln)

    def _logg(self, samples):
        logg_x = 0.5 * self.component_dist.logpdf(self.params, samples)
        if len(logg_x.shape) == 1:
            logg_x = logg_x[:,np.newaxis]
        #return logsumexp(logg_x + np.log(np.maximum(self.weights[-1], 1e-64)), axis=1)
        return logsumexp(logg_x[:, self.weights>0] + np.log(self.weights[self.weights > 0]), axis=1)

    def _sample_g(self, n):
        #samples from g^2
        g_samples = np.zeros((n, self.component_dist.d))
        #compute # samples in each mixture pair
        g_ps = (self.weights[:, np.newaxis]*self.Z*self.weights).flatten()

        # fix by GC in case there's one component and it's 0
        #if g_ps.shape[0] == 1 and g_ps[0] == 0:
        #    self.weights = np.ones(1)
        #    g_ps = np.ones(1)
        g_ps /= g_ps.sum()
        pair_samples = np.random.multinomial(n, g_ps).reshape(self.Z.shape)
        #symmetrize (will just use lower triangular below for efficiency)
        pair_samples = pair_samples + pair_samples.T
        for k in range(self.Z.shape[0]):
            pair_samples[k,k] /= 2
        #invert sigs
        #fill in samples
        cur_idx = 0
        for j in range(self.Z.shape[0]):
            for m in range(j+1):
                n_samps = pair_samples[j,m]
                g_samples[cur_idx:cur_idx+n_samps, :] = self.component_dist.cross_sample(self.params[j, :], self.params[m, :], n_samps)
                cur_idx += n_samps
        return g_samples

    def _get_mixture(self):
        #get the mixture weights
        ps = (self.weights[:, np.newaxis]*self.Z*self.weights).flatten()
        ps /= ps.sum()
        paired_params = np.zeros((self.params.shape[0]**2, self.params.shape[1]))
        for i in range(self.N):
            for j in range(self.N):
                paired_params[i*self.N + j, :] = self.component_dist._get_paired_param(self.params[i, :], self.params[j, :], flatten=True)
        #get the unflattened params and weights
        output = self.component_dist.unflatten(paired_params)
        output.update([('weights', ps)])
        return output
####################################
