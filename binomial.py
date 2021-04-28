
from math import exp, log, sqrt, pi
import numpy as np

class Binomial():
    def __init__(self, r, T, dt, S0, u, d, y=0., q=None):
        self.attr = dict()
        self.attr['r'] = r
        self.attr['T'] = T
        self.attr['dt'] = dt
        self.attr['s0'] = S0
        self.attr['y'] = y
        self.attr['u'] = u
        self.attr['d'] = d
        self.attr['q'] = q

    def price_bond(self, c):
        T = self.attr['T']
        r = self.attr['r']
        return sum([c / (1+r)**t for t in range(T)]) + (1+c) / (1+r)**(T-1)

    def price_put(self, K, type='E', dates=None, pen=0.):
        return self.price(lambda s: put(s, K), type, dates, pen)

    def price_call(self, K, type='E', dates=None, pen=0.):
        return self.price(lambda s: call(s, K), type, dates, pen)

    def price(self, payoff, type='E', dates=None, pen=0.):
        r, T, dt, S0, y, u, d, q = self.attr.values()
        if type == 'B':
            assert(len(dates) >= 1 and (T-1) in dates)

        g = exp(-r*dt)
        if q is None:
            q = (1/g - d) / (u - d)
        p = np.zeros((T, T))

        for i in range(T):
            S = S0 * u**i * d**(T-i-1) * (1 - y) ** (T-2)
            p[i, T-1] = payoff(S)

        for j in reversed(range(T-1)):
            for i in range(j+1):
                p[i, j] = g * (q*p[i+1, j+1] + (1-q)*p[i, j+1])
                # We assume that
                # S^c_{t+1} = (U|D) * S_{t}
                # So we need to remove the dividend ∂
                # S_{t+1} = S^c_{t+1} * (1 - ∂)
                S = S0 * u**i * d**(j-i) * (1 - y) ** j
                if type == 'A':
                    p[i, j] = max(payoff(S), p[i, j])
                if type == 'B' and dates is not None and j in dates:
                    p[i, j] = max(payoff(S), p[i, j])
                if type == 'G':
                    p[i, j] = max(min(p[i, j], pen+payoff(S)), payoff(S))

        return p


def call(S, K):
    return max(0, S-K)


def put(S, K):
    return max(0, K-S)
