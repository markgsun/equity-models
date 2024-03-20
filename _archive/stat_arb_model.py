# -*- coding: utf-8 -*-
"""
Created on Tue May 12 08:33:00 2020

@author: markg

Statistical arbitrage trading model for daily returns
"""

import cvxopt
import equity_util as eq
import numpy as np
import pandas as pd
from sklearn import linear_model, covariance

# Beta model
def beta_model(px_close):
    # Construct equal-weighted portfolio
    dates = px_close.index[1:]
    labels = px_close.columns
    ret = pd.DataFrame(eq.calc_return(px_close),index = dates, columns = labels)
    ret_eq = ret.mean(axis = 1)
    
    # Create beta series
    beta = pd.Series(0.0, index = px_close.columns)
    
    # Calculate beta for each stock
    for stock in ret.columns:
        # Linear
        y = ret[stock].values
        x = ret_eq.values.reshape(-1,1)
        model = linear_model.LinearRegression().fit(x,y)
        beta[stock] = model.coef_[0]
        
    return beta

# Sigma (covariance) model
def sigma_model(px_close):
    # Compute returns
    ret = eq.calc_return(px_close)
    
    # Covariance matrix
    lw = covariance.LedoitWolf().fit(ret)
    sigma = lw.covariance_
    
    # Test eigenvalues
    assert np.all(np.linalg.eigvals(sigma) > 0)
    
    return sigma

# Tau (transaction cost) model
def tau_model(px_close):
    return [0.0002]*px_close.shape[1]

# Max trade and position
def max_size(w):
    # Maximum trade and position
    if sum([abs(i) for i in w]) ==0:
        theta = 150000
        pi = 10*theta
    else:
        theta = min(150000,sum([abs(i) for i in w])*0.01)
        pi = min(10*theta, sum(np.array(w)[np.array(w)>0]))
    
    # Combine constraints
    gamma = np.maximum(np.array(w)-theta,[-pi]*len(w))
        
    delta = np.minimum(np.array(w)+theta,[pi]*len(w))
    
    return gamma, delta

# Optimize portfolio at time t
def optimize_port(w, beta, sigma, px_close, px_vol, bk2mkt, t):
    # number of stocks
    n = px_close.shape[1]
    
    # Alpha
    alpha_t = eq.alpha_model(t, px_close, px_vol, bk2mkt)
    gamma_t, delta_t = max_size(w)
    
    # Modulators
    mu = 1
    
    # CVXOPT variables
    P = cvxopt.matrix(mu*sigma)
    q = cvxopt.matrix(-alpha_t)
    
    # Equality
    A = cvxopt.matrix(np.reshape(beta.values,[1,n]))
    b = cvxopt.matrix(0, tc = 'd')
    
    # Inequalities
    G = cvxopt.matrix(np.concatenate((np.identity(n)*-1,np.identity(n))))
    h = cvxopt.matrix(np.concatenate((-gamma_t,delta_t)), tc = 'd')
    
    # Solve
    sol = cvxopt.solvers.qp(P,q,G,h,A,b)
    
    # Portfolio
    port_t = np.array(sol['x'])
    
    return port_t

# Model
def trade_model(px_close, px_vol, bk2mkt):
    # Starting time
    ti = 250
    
    # Beta
    beta = beta_model(px_close)
    sigma = sigma_model(px_close)
    
    [m,n] = px_close.shape
    
    # Full portfolio over time
    port_full = np.zeros([m-ti,n])
    port_t1 = port_full[0,:]
    
    # Optimize over time
    for t in range(ti,px_close.shape[0]):
        print(px_close.index[t])
        port_t = optimize_port(port_t1, beta, sigma, px_close, px_vol, bk2mkt, t)
        port_full[t-ti,:] = port_t.flatten()
        
        # Tests
        assert np.dot(port_t.flatten(),beta) < 0.01
        
        # Update portfolio
        port_t1 = port_t.flatten()
    
        # Construct portfolio dataframe
        dates = px_close.index[ti:]
        labels = px_close.columns
        port_full_pd = pd.DataFrame(port_full, index = dates, columns = labels)
        
    return port_full_pd

# Backtest
def backtest(px_close, port_full):
    # Calculate returns as dataframe
    labels = px_close.columns
    ret_full = pd.DataFrame(eq.calc_return(px_close),index = px_close.index[1:], columns = labels)
    
    # Backtest
    pnl = port_full_pd.iloc[:,0]*0
    for t in port_full_pd.index:
        pnl[t] = np.dot(ret_full.loc[t],port_full_pd.loc[t])
        
    # Cumulative pnl
    pnl_cum = pnl.cumsum()
    pnl_cum.plot()
    
    return pnl_cum

# Execution
if __name__ == '__main__':
    # Silence cvxopt
    cvxopt.solvers.options['show_progress'] = False
    
    # Parameters
    start = '2017-01-01' 
    end = '2020-05-01'
    idx = 'S&P 500'
    
    # Pull data
    px_close = eq.pull_hist('Close', start, end, idx = idx)
    px_vol = eq.pull_hist('Volume', start, end, idx = idx)
    bk2mkt = eq.pull_bk2mkt(start, end, px_close, idx = idx)
    
    # Full portfolio over time
    port_full_pd = trade_model(px_close, px_vol, bk2mkt)
    
    # Calculate returns as dataframe
    pnl_cum = backtest(px_close, port_full_pd)
    
    