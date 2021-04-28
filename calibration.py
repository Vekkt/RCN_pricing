import csv
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import minimize
from binomial import Binomial
import matplotlib.pyplot as plt

def calibration():
    P = []
    C = []
    K = []
    S0 = 11118

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

    y = pd.Series(P+S0-C, name='y')
    data = pd.concat([K, y], axis=1)

    results = smf.ols('y ~ K', data=data).fit()
    b1, b2 = results.params
    y, r = b1 / S0, -np.log(b2)
    print(("Interest rate : {:.4f}\n"
           "Dividend yield : {:.4f}").format(r, y))

    def f(x):
        u, d = x
        tree = Binomial(r/12, T, dt, I0, u, d, y/12)
        prices = pd.Series([tree.price_call(k) for k in K], name='sim_c')
        return ((C-prices)**2).sum()

    T  = 12
    dt = 1/12
    I0 = 11118

    cons = ({'type': 'ineq', 'fun': lambda x: x[0] - np.exp(r*dt)},
            {'type': 'ineq', 'fun': lambda x: np.exp(r*dt) - x[1]})

    res = minimize(f, (1.1, 0.9), method='COBYLA', constraints=cons)
    if res.success:
        print('Minimum found!')
        print('Function value : {:.4f}'.format(res.fun))
        u, d = res.x
        print(("Up : {:.10f}\n"
           "Down : {:.10f}").format(u, d))

    #tree = Binomial(r/12, T, dt, I0, u, d, y/12)
    #sim = pd.Series([tree.price_call(k) for k in K], name='sim_c')
    #sim = pd.concat([sim, C], axis=1).set_index(K.values)
    #print(sim)
    return r, y, u, d

if __name__ == '__main__':
    calibration()