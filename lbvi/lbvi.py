# suite of functions for doing locally-adapted boosting variational inference

# preamble
import numpy as np
import scipy.stats as stats
import cvxpy as cp
import time, bisect
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import cProfile, pstats, io
