#%%
from math import exp, log, sqrt, pi
import numpy as np
from rcn import rcn
from binomial import Binomial
from calibration import calibration


T     = 12
dt    = 1/T
i0    = 11118
c     = 0.1
alpha = 1
beta  = 0.8

r, y, u, d = calibration()
tree = Binomial(r, T, dt, i0, u, d, y)

print('{:10} : {:.4f}'.format('  rcn', tree.price_RCN(alpha, c)))
print('{:10} : {:.4f}'.format(' brcn', tree.price_RCN(alpha, c, beta)))

print('{:10} : {:.4f}'.format(' crcn', tree.price_RCN(alpha, c, callable=True)))
print('{:10} : {:.4f}'.format('cbrcn', tree.price_RCN(alpha, c, beta, callable=True)))


print()
note = rcn(r, dt, i0, y, u, d, c, T)
print('{:10} : {:.4f}'.format(' rcn', note.price_rcn(alpha=alpha, c=c)))
print('{:10} : {:.4f}'.format(
    'brcn', note.price_brcn(alpha=alpha, beta=beta, c=c)))
dates = [j for j in range(1, T)]
RCN, rep1 = note.price_rcn(
    alpha=alpha, c=c, dates=dates, replication_strategy=True)
BRCN, rep2 = note.price_brcn(
    alpha=alpha, c=c, beta=0.8, dates=dates, replication_strategy=True)
print('{:10} : {:.4f}\n{:10} : {:.4f}'.format('crcn',
    RCN, 'cbrcn', BRCN))
