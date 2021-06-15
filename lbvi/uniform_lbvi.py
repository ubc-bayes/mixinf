# suite of functions for doing locally-adapted boosting variational inference
# with uniform step size increase

# preamble
import numpy as np
import scipy.stats as stats
import pandas as pd
import time, bisect
plt.rcParams.update({'figure.max_open_warning': 0})
import cProfile, pstats, io
import os
import imageio
from lbvi import mix_sample, ksd, kl, simplex_project, plotting, gif_plot, w_grad, weight_opt,
