# -*- coding: utf-8 -*-
"""
Created on Tue May 12 08:33:00 2020

@author: markg
"""

import datetime as dt
import numpy as np
import cvxopt
import pandas as pd
from sqlalchemy import create_engine
from sklearn import linear_model, covariance
from urllib import parse

# Pull data from database

# Establish connection to database
def sqlconn(db):
    params = parse.quote_plus('DRIVER={SQL Server};'
                              'SERVER=DESKTOP-9HGRDTD\SQLEXPRESS;'
                              'DATABASE='+db+';'
                              'Trusted_Connection=yes')

    engine = create_engine('mssql+pyodbc:///?odbc_connect={}'.format(params))
    return engine

# Pull data
def pull_hist(val):
    # Connect to database
    engine = sqlconn('FinancialData')
    
    # Execute SQL statement
    res = engine.execute('SELECT * FROM EquityPx')
    data_raw = pd.DataFrame(res.fetchall())
    data_raw.columns = res.keys()
    
    # Get closing price
    px_long = data_raw[[val,'Stock','Date']]
    
    # Reshape price data
    px_wide = px_long.pivot(values = val,index = 'Date',columns = 'Stock')
    
    # Convert index to dates
    px_wide.index = [dt.datetime.strptime(date, '%b %d, %Y').date() for date in px_wide.index.values]
    
    # Sort by date
    px_wide = px_wide.sort_index()
    
    return px_wide

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
    assert abs(out.mean())<0.001
    assert abs(out.std()-1)<0.001
    assert max(abs(out))-3<0.001
    
    return out

# Straight momentum alpha
def alpha_mom_str(data_period):
    data_diff = data_period.iloc[[-250,-1],:]
    
    # Cumulative return
    ret = data_diff.iloc[1,:]/data_diff.iloc[0,:]-1
    
    # Windsorize
    a_mom = windsorize(ret)
    
    return a_mom

# Weekly volume alpha
def alpha_vol_wk(data_period):
    data_1wk = data_period.iloc[-6:-1,:]
    
    # Average volume
    avg_vol = data_1wk.mean(axis = 0)
    
    # Windsorize
    a_vol = windsorize(avg_vol)
    
    return a_vol

# Alpha model
def alpha_model(t, px_close, px_vol):
    # Calculate individual alphas
    a_mom = alpha_mom_str(px_close.iloc[0:t,:])
    a_vol = alpha_vol_wk(px_vol.iloc[0:t,:])
    
    # Aggregate alphas
    wts = [0.5,0.5]
    alphas = pd.concat([a_mom,a_vol], axis = 1)*wts
    alpha = alphas.sum(axis = 1)
    
    return alpha

# Beta model
def beta_model(px_close):
    # Construct equal-weighted portfolio
    ret = px_close.iloc[1:,:]/px_close.iloc[0:-1,:].values-1
    ret_eq = ret.mean(axis = 1)
    
    # Remove nan
    ret = ret.fillna(0)
    
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
    ret = px_close.iloc[1:,:]/px_close.iloc[0:-1,:].values-1
    
    # Remove nan
    ret = ret.fillna(0)
    
    # Covariance matrix
    lw = covariance.LedoitWolf().fit(ret)
    sigma = lw.covariance_
    
    # Test eigenvalues
    assert np.all(np.linalg.eigvals(sigma) > 0)
    
    return sigma

# Tau (transaction cost) model
def tau_model(px_close):
    return [0.0002]*px_close.shape[1]
# Execution
if __name__ == '__main__':
    # Pull data
    px_close = pull_hist('Close')
    px_vol = pull_hist('Volume')
    
    # Time index
    t = 250
    n = px_close.shape[1]
    
    # Alpha
    alpha_t = alpha_model(t, px_close, px_vol)
    
    # Beta
    beta = beta_model(px_close)
    
    # Sigma (Covariance)
    sigma = sigma_model(px_close)
    
    # Tau (Transaction Cost)
    tau = tau_model(px_close)
    
    # Variables
    lamb = 1
    mu  = 1
    pi = 100000
    theta = [10000]*n
    
    
    
    