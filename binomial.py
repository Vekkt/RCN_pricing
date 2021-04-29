
from math import exp, log, sqrt, pi
import numpy as np

np.set_printoptions(threshold=np.inf, suppress=True, linewidth=100000)


class Binomial():
    def __init__(self, r, T, dt, S0, u, d, y=0., q=None):
        self.attr = dict()  # easier to unpack attributes later with a dict
        self.attr['r'] = r
        self.attr['T'] = T
        self.attr['dt'] = dt
        self.attr['s0'] = S0
        self.attr['y'] = y
        self.attr['u'] = u
        self.attr['d'] = d
        self.attr['q'] = q
        self.attr['ind'] = []
        self.attr['asset'] = []

        self.attr['asset'] = self.underlying_price(beta)

    def price_bond(self, c):
        T = self.attr['T']
        r = self.attr['r']
        # We assume continuous compounding
        return sum([c * exp(-r * dt * t) for t in range(1, T)]) + (1+c) * exp(-r * dt * T)

    def price_put(self, K, type='E', dates=None, pen=0.):
        return self.price(lambda s: put(s, K), type, dates, pen)

    def price_call(self, K, type='E', dates=None, pen=0.):
        return self.price(lambda s: call(s, K), type, dates, pen)

    def price_barrier_call(self, K, beta, type='KI'):
        return self.price_barrier(lambda s: call(s, K), beta, type)

    def price_barrier_put(self, K, beta, type='KI'):
        return self.price_barrier(lambda s: put(s, K), beta, type)

    def price(self, payoff, type='E', dates=None, pen=0.):
        r, T, dt, S0, y, u, d, q, _, _ = self.attr.values()
        if type == 'B':
            assert(len(dates) >= 1 and (T-1) in dates)

        g = exp(-r*dt)
        if q is None:
            q = (1/g - d) / (u - d)
        p = np.zeros((T, T))

        for i in range(T):
            S = S0 * u**(T-i-1) * d**i * (1 - y) ** (T-2)
            p[i, T-1] = payoff(S)

        for j in reversed(range(T-1)):
            for i in range(j+1):
                p[i, j] = g * (q*p[i, j+1] + (1-q)*p[i+1, j+1])
                # We assume that
                # S^c_{t+1} = (U|D) * S_{t}
                # So we need to remove the dividend ∂
                # S_{t+1} = S^c_{t+1} * (1 - ∂)
                S = S0 * u**(j-i) * d**i * (1-y)**j
                if type == 'A':  # American
                    p[i, j] = max(payoff(S), p[i, j])
                if type == 'B' and dates is not None and j in dates:  # Bermudan
                    p[i, j] = max(payoff(S), p[i, j])
                if type == 'G':  # Game
                    p[i, j] = max(min(p[i, j], pen+payoff(S)), payoff(S))
        return p

    def underlying_price(self, beta=None):
        r, T, dt, S0, y, u, d, q, _, _ = self.attr.values()

        g = exp(-r*dt)
        if q is None:
            q = (1/g - d) / (u - d)

        if beta is not None:
            ind = []
            # If the asset starts below the barrier, this is an Up option
            if beta > 1:
                type = 'U'
            # If the asset starts below the barrier, this is a Down option
            if beta <= 1:
                type = 'D'

        s_ex = np.zeros((2**(T-1), T))
        s_ex[0, 0] = S0

        # We populate the tree forward.
        # From (i,j) to (2*i, j+1) and (2*i+1, j+1)
        for j in range(T-1):
            for i in range(2**j):
                # We assume that
                # S^c_{t+1} = (U|D) * S_{t}
                # So we need to remove the dividend ∂
                # S_{t+1} = S^c_{t+1} * (1 - ∂)
                s_ex[2*i, j+1] = u * s_ex[i, j] * (1-y)
                s_ex[2*i+1, j+1] = d * s_ex[i, j] * (1-y)

                if beta is not None:
                    # We add all leaves that come from a node
                    # where the threshold has been breached
                    if (type == 'D' and s_ex[i, j] <= beta * S0 or
                            type == 'U' and s_ex[i, j] >= beta * S0):
                        # L is the number of leaves from this node
                        # i * L is the index of the first leave
                        L = 2**(T-1-j)
                        ind.extend([i * L + k for k in range(L)])

        if beta is not None:
            # remove duplicates and save
            self.attr['ind'] = list(set(ind.copy()))

        return s_ex

    def price_barrier(self, payoff, beta, type='KI'):
        r, T, dt, _, _, u, d, q, ind, underlying = self.attr.values()

        g = exp(-r*dt)
        if q is None:
            q = (1/g - d) / (u - d)

        p = np.zeros_like(underlying)

        for i in range(2**(T-1)):
            p[i, T-1] = payoff(underlying[i, T-1])
            if type[1] == 'I' and i not in ind:
                p[i, T-1] = 0
            elif type[1] == 'O' and i in ind:
                p[i, T-1] = 0

        for j in reversed(range(T-1)):
            for i in range(2**j):
                p[i, j] = g * (q*p[2*i, j+1] + (1-q)*p[2*i+1, j+1])
        return p

    def price_RCN(self, alpha, c, beta=None, callable=False):
        S0 = self.attr['s0']
        K = alpha*S0
        # Basic RCN
        if beta is None and not callable:
            bond = self.price_bond(c)
            put = self.price_put(K)[0, 0]
            return bond - put / S0

        if beta is not None and not callable:
            bond = self.price_bond(c)
            ki_put = self.price_barrier_put(K, beta, type='KI')[0, 0]
            return bond - ki_put / S0


def call(S, K):
    return max(0, S-K)


def put(S, K):
    return max(0, K-S)


if __name__ == '__main__':
    T = 12
    dt = 1/T
    r = 0.02
    S0 = 100
    u = 2
    d = 0.5
    y = 0.01
    k = 100
    c = 0.1
    K = 100
    alpha = 0.9
    beta = 0.8

    tree = Binomial(r, T, dt, S0, u, d)
    print('{:15} : {:.4f}'.format('rcn', tree.price_RCN(alpha, c)))
    print('{:15} : {:.4f}'.format('brcn', tree.price_RCN(alpha, c, beta)))