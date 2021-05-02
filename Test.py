#%%
from math import exp, log, sqrt, pi
import numpy as np
from rcn import rcn
from binomial import Binomial
from calibration import calibration



# r = -0.0078
# u = 1.0826
# d = 0.9685
# y = 0.0278
# c = 0.1
# alpha = 1

T = 4
dt = 1/T
S0 = 11118
K = 10_000
beta = 0.9
r, y, u, d = calibration()
tree = Binomial(r, T, dt, S0, u, d, y)

verbose = True
tree.price_barrier_put(K, beta, type='KI', verbose=verbose)
