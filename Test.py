#%%
from math import exp, log, sqrt, pi
import numpy as np
from rcn import rcn
from binomial import Binomial



T = 12
dt = 1/12
r = -0.0078
S0 = 100
u = 1.0826
d = 0.9685
y = 0.0278
k = 100
c = 0.1
alpha = 1
beta = 0.8

tree = Binomial(r, T, dt, S0, u, d, y, q=None)

verbose = True
print('{:10} : {:.6f}'.format(' rcn', tree.price_RCN(alpha, c*dt, verbose=False)))
print('{:10} : {:.6f}'.format('brcn', tree.price_RCN(alpha, c*dt, beta, verbose=False)))
print('{:10} : {:.6f}'.format('bond', tree.price_bond(c*dt)))


rcn_tree = rcn(r, dt, S0, y, u, d, c, T, q=None)
print('\n{:10} : {:.6f}'.format(' rcn', rcn_tree.price_rcn(alpha, c)))
print('{:10} : {:.6f}'.format('brcn', rcn_tree.price_brcn(alpha, beta, c*dt)))
print('{:10} : {:.6f}'.format('bond', rcn_tree.bond))

