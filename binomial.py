
import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000)

from math import exp, log, sqrt, pi

class Binomial():
    def __init__(self, r, T, dt, S0, u, d, y=0.):
        self.attr = dict()
        self.attr['r']  = r
        self.attr['T']  = T
        self.attr['dt'] = dt
        self.attr['s0'] = S0
        self.attr['y']  = y
        self.attr['u']  = u
        self.attr['d'] = d

    def price_bond(self, c):
        T = self.attr['T']
        r = self.attr['r']
        b = sum([c * exp(-r * dt * t) for t in range(1, T)]) + (1 + c) * exp(- r * dt * T)
        return b

    def price_put(self, K, type='E', dates=None, pen=0.):
        return self.price(lambda s: put(s, K), type, dates, pen)[0,0]

    def price_call(self, K, type='E', dates=None, pen=0.):
        return self.price(lambda s: call(s, K), type, dates, pen)[0,0]

    def price(self, payoff, type='E', dates=None, pen=0.):
        r, T, dt, S0, y, u, d = self.attr.values()
        if type == 'B': assert(len(dates) >= 1 and (T-1) in dates)

        g  = exp(-r*dt)
        #q  = (1/g - d) / (u - d)
        q = 0.4521255414041359
        p  = np.zeros((T, T))

        for i in range(T):
            S = S0 * u**i * d**(T-i-1) * (1 - y) ** (T-2)
            p[i, T-1] = payoff(S)

        for j in reversed(range(T-1)):
            for i in range(j+1):
                p[i,j] = g * (q*p[i+1, j+1] + (1-q)*p[i, j+1])
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

if __name__ == '__main__':
    T  = 4
    dt = 1/12
    r  = 0.02
    I0 = 100
    u  = 1.100
    d  = 0.919
    y  = 0.
    k = 100
    c = 0.1

    tree = Binomial(r,T,dt,I0,u,d)

    #print(tree.price_bond(c))
    print(tree.price_bond(c) - tree.price_put(k, 'A')/I0)



