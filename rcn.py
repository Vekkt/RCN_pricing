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
    def __init__(self, r, dt, i0, div, u, d):
        self.r = r #interest rate
        self.dt = dt    #time increment
        self.i0 = i0    #initial value of the underlying
        self.div = div  #dividend yield
        self.u = u
        self.d = d
        self.t_end = int(1 / self.dt)   # maturity date in numbers of increment
        #self.bond = sum(c * exp(-r * dt * t) for t in range(1, self.t_end))\
        #            + (1 + c)*exp(-r*dt*self.t_end) #bond price,never used
        self.q = 1/2

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
        s_cum = s_ex.copy()


        for j in range(1, T + 1):
            for i in np.nonzero(s_ex[:, j - 1])[0]:
                s_ex[2 * i, j] = s_ex[i, j - 1] * u * (1 - y)
                s_ex[2 * i + 1, j] = s_ex[i, j - 1] * d * (1 - y)

                if beta:
                    if s_ex[i, j - 1] <= beta * i0:
                        self.ind.extend([2**(T - (j - 1)) * i + n for n in range(2**(T - (j-1)))])
                        #if the stock price is below the barrier then all the state this node leads to in the last period
                        #will be stored

        self.ind = set(self.ind)    #set of indicies that hit the barrier

        return s_ex



    def payoff(self, s, alpha):
        return 1 - np.maximum(0, alpha - s / self.i0)

    def price_rcn(self, alpha, c, dates=None):
        T = self.t_end
        r = self.r
        dt = self.dt
        s = self.stock_tree()   #call the underlying tree
        q = self.q #prob of going up under Q

        rcn = np.zeros([2 ** T, T + 1]) #initialize the matrix
        rcn[:, -1] = c + self.payoff(s[:, -1], alpha) #payoff at maturity

        if dates:
            self.callable = True

        for j in range(1, T + 1):
             for i in range(2 ** (T - j)):
                rcn[i, T - j] = np.exp(-r * dt) * (rcn[2 * i, T - j + 1] * q + (1 - q) * rcn[2 * i + 1, T - j + 1]) + c
                #cum dividend price at t is the expectation of the cum dividend prices at t+1 added the coupon payment
                if callable and dates is not None and j in dates:
                    rcn[i, T - j] = min(rcn[i, T - j], 1 + c) # if RCN is callable select the minimum of the continuation price and full repayement

        return rcn[0, 0] - c   #subtract the coupon payment at date t=0 since no coupon is paid at this date


    def price_brcn(self, alpha, beta, c, dates=None):
        T = self.t_end
        q = self.q
        r = self.r
        dt = self.dt
        s = self.stock_tree(beta)   #call the underlying tree

        brcn = np.zeros([2 ** T, T + 1]) #initialize the matrix
        brcn[:, -1] = c + self.payoff(s[:, -1], alpha)    #payoff with above strike

        if dates:
            self.callable = True

        universal = set(i for i in range(2 ** T)) #set of all indicies
        below_strike = set(np.argwhere(brcn[:, -1] < 1 + c).reshape(-1))   # set of indicies below strike at maturity
        mid = list((universal - self.ind) & below_strike)   #set of indicies that did not hit barrier and below strike
        brcn[mid, -1] = 1 + c  # payoff of paths that ended below

        for j in range(1, T + 1):
            for i in range(2 ** (T - j)):
                brcn[i, T - j] = np.exp(-r * dt) * (brcn[2 * i, T - j + 1] * q + (1 - q) * brcn[2 * i + 1, T - j + 1]) + c
                #cum dividend price at t is the expectation of the cum dividend prices at t+1 added the coupon payment

                if callable and dates is not None and j in dates:
                    brcn[i, T - j] = min(brcn[i, T - j], 1 + c) # if BRCN is callable select the minimum of the continuation price and full repayement

        return brcn[0, 0] - c  #subtract the coupon payment at date t=0 since no coupon is paid at this date


    def recomb_rcn(self, alpha, c, dates=None):
        T = self.t_end
        q = self.q
        u = self.u
        d = self.d
        y = self.div
        c = self.c
        r = self.r
        dt = self.dt
        i0 = self.i0

        if dates:
            self.callable = True

        rcn = np.zeros([T + 1, T + 1])  #initialize payoff matrix
        for i in range(T + 1):
            rcn[T - i, -1] = i0 * (u * (1 - y)) ** i * (d * (1 - y)) ** (T - i) #populate the last period

        rcn[:, -1] = c + (1 - np.maximum(0, alpha - rcn[:, -1] / i0))  # payoff at maturity

        for j in reversed(range(T)):
            for i in range(j + 1):
                rcn[i, j] = np.exp(-r * dt) * ( rcn[i + 1, j + 1] * (1-q)
                                                                       + q * rcn[i, j + 1]) + c

                if callable and dates is not None and j in dates:
                    rcn[i, j] = min(rcn[i, j], 1 + c) # if RCN is callable select the minimum of the continuation price and full repayement
        return rcn[0, 0] - c




if __name__ == '__main__':
    T  = 1
    dt = 1/12
    r  = 0.02
    I0 = 100
    u  = 2
    d  = 0.5
    y  = 0.
    k = 100
    div = 0.01
    c = 0.1
    dates = [i for i in range(int(T/dt)+1)]
    #dates = []
    alpha = 0.9
    beta = 0.8
    tree = rcn(r, dt, I0, div, u, d, c)
    #s_ex, s_cum = tree.stock_tree()
    brcn = tree.price_brcn(alpha=alpha, beta=beta, dates=dates)
    rcn = tree.price_rcn(alpha=alpha, dates=dates)
    recomb_rcn = tree.recomb_rcn(alpha=alpha, dates=dates)
    #print('brcn=', brcn)
    print(' rcn=', rcn)
    print('recomb rcn=', recomb_rcn)
    print('brcn=', brcn)