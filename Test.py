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
c = 0.15
alpha = 1.1

T = 12
dt = 1/T
i0 = 11118
K = 10_000
beta = 1.11
r, y, u, d = calibration()
tree = Binomial(r, T, dt, i0, u, d, y)

verbose = False
tree.price_barrier_put(K, beta, type='KI', verbose=verbose)

tree = Binomial(r, T, dt, i0, u, d, y)

print('{:10} : {:.4f}'.format(' rcn', tree.price_RCN(alpha, c)))
print('{:10} : {:.4f}'.format('brcn', tree.price_RCN(alpha, c, beta)))
print('{:10} : {:.4f}'.format('bond', tree.price_bond(c*dt)))

print()
note = rcn(r, dt, i0, y, u, d, c, T)
print('{:10} : {:.4f}'.format(' rcn', note.price_rcn(alpha=alpha, c=c)))
print('{:10} : {:.4f}'.format('brcn', note.price_brcn(alpha=alpha, beta=beta, c=c)))
print('{:10} : {:.4f}'.format('bond', note.bond))