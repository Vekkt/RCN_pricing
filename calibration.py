import csv
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import minimize
from binomial import Binomial
import matplotlib.pyplot as plt
from math import exp

def get_data():
    P, C, K = [], [], []

    with open('Data-Project1-Fin404.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            try:
                K.append(int(row[0]))
                C.append(float(row[1]))
                P.append(float(row[2]))
            except:
                pass

    K = pd.Series(K, name='K')
    C = pd.Series(C, name='Call')
    P = pd.Series(P, name='Put')

    return P, C, K

def calibration():
    # Fetch data
    ###########################################################################

    P, C, K = get_data()
    S0 = 11118

    # Estimate the dividend yield and risk-free rate
    # Using put/call parity
    ###########################################################################

    y = pd.Series(P+S0-C, name='y')
    data = pd.concat([K, y], axis=1)

    results = smf.ols('y ~ K', data=data).fit()
    b1, b2 = results.params
    y, r = -np.log(1-b1 / S0) / 12, -np.log(b2)
    print('{:15}{:.4f}'.format('Interest rate', r))
    print('{:15}{:.4f}'.format('Dividend yield', y))
    print('-'*30)


    # Calibrate the model by minimizing the square error
    # For call and put prices
    ###########################################################################

    # Optimize using call prices
    def f(x):
        u, d = x
        tree = Binomial(r, T, dt, I0, u, d, y, q=.5)
        prices = pd.Series([tree.price_call(k)[0, 0] for k in K], name='simc')
        return ((C-prices)**2).sum()

    # Optimize using put prices
    def g(x):
        u, d = x
        tree = Binomial(r, T, dt, I0, u, d, y, q=.5)
        prices = pd.Series([tree.price_put(k)[0, 0] for k in K], name='simp')
        return ((P-prices)**2).sum()

    # Sum of the two functions
    def h(x):
        return f(x) + g(x)


    T = 12
    dt = 1/12
    I0 = 11118

    cons = ({'type': 'ineq', 'fun': lambda x: x[0] - np.exp(r*dt)},
            {'type': 'ineq', 'fun': lambda x: np.exp(r*dt) - x[1]},
            {'type': 'eq', 'fun': lambda x: (exp(r*dt) - x[1]) / (x[0] - x[1]) - 0.5})

    res = minimize(h, (1.0832, 0.9678), constraints=cons)

    if res.success:
        u, d = res.x
        print('Minimum found!')
        print('{:15}{:.4f}'.format('Func value', res.fun))
        print('{:15}{:.4f}'.format('Up', u))
        print('{:15}{:.4f}'.format('Down', d))
        print('-'*30)

    tree = Binomial(r, T, dt, I0, u, d, y, q=.5)

    simc = pd.Series([tree.price_call(k)[0, 0] for k in K], name='simc')
    simc = pd.concat([simc, C], axis=1).set_index(K.values)
    simp = pd.Series([tree.price_put(k)[0, 0] for k in K], name='simp')
    simp = pd.concat([simp, P], axis=1).set_index(K.values)

    sim = pd.concat([simc, simp], axis=1)
    sim.index.name = 'Strike'
    print(sim)
    plt.plot(sim)
    plt.show()
    return r, y, u, d

if __name__ == '__main__':
    calibration()
