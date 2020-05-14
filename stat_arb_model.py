# -*- coding: utf-8 -*-
"""
Created on Tue May 12 08:33:00 2020

@author: markg

Statistical arbitrage trading model for daily returns
"""

import cvxopt
import equity_shared as eq
import numpy as np
import pandas as pd
from sklearn import linear_model, covariance

def windsorize(raw):
    ct = 0
    out = raw
    while(ct < 10):
        # Standardize
        out = (out-out.mean())/out.std()
        # Windsorize
        out[abs(out)>3] = 3*out[abs(out)>3]/abs(out[abs(out)>3])
        ct += 1
    
    # Tests
    assert abs(out.mean())<0.01
    assert abs(out.std()-1)<0.01
    assert max(abs(out))-3<0.01
    
    return out

# Straight momentum alpha
def alpha_mom_str(data_period):
    # Cumulative return
    ret = data_period.iloc[-1,:]/data_period.iloc[0,:]-1
    
    # Windsorize
    a_mom = windsorize(ret)
    
    return a_mom

# Weekly volume alpha
def alpha_vol_wk(data_period):
    data_1wk = data_period.iloc[-5:-1,:]
    
    # Average volume
    avg_vol = data_1wk.mean(axis = 0)
    
    # Windsorize
    a_vol = windsorize(avg_vol)
    
    return a_vol

# Alpha model
def alpha_model(t, px_close, px_vol):
    # Calculate individual alphas
    a_mom = alpha_mom_str(px_close.iloc[t-60:t,:])
    a_vol = alpha_vol_wk(px_vol.iloc[t-5:t,:])
    
    # Aggregate alphas
    wts = [0,1]
    alphas = pd.concat([a_mom,a_vol], axis = 1)*wts/sum(wts)
    alpha = alphas.sum(axis = 1)
    
    return alpha

# Beta model
def beta_model(px_close):
    # Construct equal-weighted portfolio
    dates = px_close.index[1:]
    labels = px_close.columns
    ret = pd.DataFrame(calc_return(px_close),index = dates, columns = labels)
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
    ret = calc_return(px_close)
    
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
def optimize_port(w, beta, sigma, px_close, px_vol, t):
    # number of stocks
    n = px_close.shape[1]
    
    # Alpha
    alpha_t = alpha_model(t, px_close, px_vol)
    gamma_t, delta_t = max_size(w)
    
    # CVXOPT variables
    P = cvxopt.matrix(sigma)
    q = cvxopt.matrix(alpha_t)
    
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
def trade_model(px_close, px_vol):
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
        port_t = optimize_port(port_t1, beta, sigma, px_close, px_vol, t)
        port_full[t-ti,:] = port_t.flatten()
        
        # Tests
        assert np.dot(port_t.flatten(),beta) < 0.01
        
        # Update portfolio
        port_t1 = port_t
    
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
    
    # Pull data
    px_close = eq.pull_hist('Close')
    px_vol = eq.pull_hist('Volume')
    
    # Full portfolio over time
    port_full_pd = trade_model(px_close, px_vol)
    
    # Calculate returns as dataframe
    pnl_cum = backtest(px_close, port_full_pd)
    
    