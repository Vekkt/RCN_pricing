import numpy as np
from math import exp, log, sqrt, pi


class rcn():
    """Specify intereest rate r
    discretization period dt
    initial index price i0
    dividiend yield
    up factor u
    down factor d
    coupon rate c"""

    def __init__(self, r, dt, i0, div, u, d, c, T, q=None):
        self.r = r  # interest rate
        self.dt = dt  # time increment
        self.i0 = i0  # initial value of the underlying
        self.div = div  # dividend yield
        self.u = u
        self.d = d
        self.t_end = T   # maturity date in numbers of increment
        c = c * dt
        self.bond = sum(c * exp(-r * dt * t) for t in range(1, self.t_end)) + (1 + c)*exp(-r*dt*self.t_end)  # bond price,never used
        if q is None:
            q = (exp(r * dt) - d) / (u-d)
        self.q = q


    def stock_tree(self, beta=0):
        """Simulates the underlying stock"""
        T = self.t_end
        u = self.u
        d = self.d
        y = self.div
        i0 = self.i0
        self.ind = []

        s_ex = np.zeros([2 ** T, T + 1])
        s_ex[0, 0] = self.i0

        for j in range(1, T + 1):
            for i in np.nonzero(s_ex[:, j - 1])[0]:
                s_ex[2 * i, j] = s_ex[i, j - 1] * u * (1 - y)
                s_ex[2 * i + 1, j] = s_ex[i, j - 1] * d * (1 - y)

                if beta:
                    if s_ex[i, j - 1] <= beta * i0: #I'm changing j-1 to j
                        self.ind.extend(
                            [2**(T - (j - 1)) * i + n for n in range(2**(T - (j - 1)))])
                        #if the stock price is below the barrier then all the state this node leads to in the last period
                        #will be stored
        if beta:
            for i in range(2**T):
                if s_ex[i, T] <= beta * i0:  # I'm changing j-1 to j
                    self.ind.append(i)

        self.ind = set(self.ind)  # set of indicies that hit the barrier

        return s_ex

    def payoff(self, s, alpha):
        return 1 - np.maximum(0, alpha - s / self.i0)

    def price_rcn(self, alpha, c, dates=None):
        T = self.t_end
        r = self.r
        dt = self.dt
        s = self.stock_tree()  # call the underlying tree
        q = self.q  # prob of going up under Q
        rcn = np.zeros([2 ** T, T + 1])  # initialize the matrix
        c = c * dt #correct for annual rate
        rcn[:, -1] = c + self.payoff(s[:, -1], alpha)  # payoff at maturity
        #(s)
        if dates:
            self.callable = True

        for j in range(1, T + 1):
            for i in range(2 ** (T - j)):
                rcn[i, T - j] = np.exp(-r * dt) * (rcn[2 * i, T - j + 1]
                                                   * q + (1 - q) * rcn[2 * i + 1, T - j + 1]) + c
                #cum dividend price at t is the expectation of the cum dividend prices at t+1 added the coupon payment
                if callable and dates is not None and j in dates:
                    # if RCN is callable select the minimum of the continuation price and full repayement
                    rcn[i, T - j] = min(rcn[i, T - j], 1 + c)
                    # if 1 + c > rcn[i, T - j]:
                    #     print('Exercise early')
                    # else:
                    #     print('Continue')
        # subtract the coupon payment at date t=0 since no coupon is paid at this date

        return rcn[0, 0] - c

    def price_brcn(self, alpha, beta, c, dates=None):
        T = self.t_end
        q = self.q
        r = self.r
        dt = self.dt
        s = self.stock_tree(beta)  # call the underlying tree
        c = c * dt #correct for annual rate

        brcn = np.zeros([2 ** T, T + 1])  # initialize the matrix
        # payoff with above strike
        brcn[:, -1] = c + self.payoff(s[:, -1], alpha)

        if dates:
            self.callable = True

        universal = set(i for i in range(2 ** T))  # set of all indicies
        # set of indicies below strike at maturity
        below_strike = set(np.argwhere(brcn[:, -1] < 1 + c).reshape(-1))
        # set of indicies that did not hit barrier and below strike
        mid = list((universal - self.ind) & below_strike)
        brcn[mid, -1] = 1 + c  # payoff of paths that ended below

        for j in range(1, T + 1):
            for i in range(2 ** (T - j)):
                brcn[i, T - j] = np.exp(-r * dt) * (brcn[2 * i, T - j + 1]
                                                    * q + (1 - q) * brcn[2 * i + 1, T - j + 1]) + c
                #cum dividend price at t is the expectation of the cum dividend prices at t+1 added the coupon payment

                if callable and dates is not None and j in dates:
                    # if BRCN is callable select the minimum of the continuation price and full repayement
                    brcn[i, T - j] = min(brcn[i, T - j], 1 + c)

        # subtract the coupon payment at date t=0 since no coupon is paid at this date
        return brcn[0, 0] - c

    def recomb_rcn(self, alpha, c, dates=None):
        T = self.t_end
        q = self.q
        u = self.u
        d = self.d
        y = self.div
        r = self.r
        dt = self.dt
        i0 = self.i0
        c = c * dt #correct for annual rate

        if dates:
            self.callable = True

        rcn = np.zeros([T + 1, T + 1])  # initialize payoff matrix
        for i in range(T + 1):
            # populate the last period
            rcn[T - i, -1] = i0 * (u * (1 - y)) * i * (d * (1 - y)) * (T - i)

        # payoff at maturity
        rcn[:, -1] = c + (1 - np.maximum(0, alpha - rcn[:, -1] / i0))

        for j in reversed(range(T)):
            for i in range(j + 1):
                rcn[i, j] = np.exp(-r * dt) * (rcn[i + 1, j + 1] * (1-q)
                                               + q * rcn[i, j + 1]) + c

                if callable and dates is not None and j in dates:
                    # if RCN is callable select the minimum of the continuation price and full repayement
                    rcn[i, j] = min(rcn[i, j], 1 + c)
        return rcn[0, 0] - c


if __name__ == '__main__':
    T = 12
    dt = 1 / 12
    r = -0.0078
    S0 = 100
    u = 1.0826
    d = 0.9685
    y = 0.0278
    k = 100
    c = 0.
    K = 100
    from binomial import Binomial

    dates = [i for i in range(int(T/dt)+1)]
    #dates = []
    note = rcn(r, dt, S0, y, u, d, c, T=12, q=None)
    T = 12
    alpha = 1
    c = 0.1
    beta = 0.8
    dates = [j for j in range(1, T)]
    RCN = note.price_rcn(alpha=alpha, c=c, dates=dates)
    BRCN = note.price_brcn(alpha=alpha, c=c, beta=0.8, dates=dates)
    print('Price of callable simple RCN {:.6f}\nPrice of callable barrier RCN {:.6f}'.format(RCN, BRCN))
    print(note.price_rcn(alpha, c))
    print(note.price_brcn(alpha, beta, c))

