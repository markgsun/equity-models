# -*- coding: utf-8 -*-
"""
Created on Thu May 14 11:04:32 2020

@author: markg

Long-only investment model for medium horizons
"""

import numpy as np
import pandas as pd
import equity_shared as eq

def split_bin(arr,n_bins):
    # Sort array
    arr_sort = pd.DataFrame(arr).sort_values(0)[10:-10]
    
    # Divide into buckets
    bin_labels = range(n_bins)
    arr_sort['Bin'] = pd.qcut(arr_sort.values.flatten(), n_bins, labels = bin_labels)
    
    return arr_sort

def bin_portfolios(px_close, alpha, n_bins):
    # Calculate returns
    ret_full = pd.DataFrame(eq.calc_return(px_close), columns = px_close.columns, index = px_close.index[1:])
    
    # Divide into bins
    alpha_bin = split_bin(alpha, n_bins)
    
    # Construct portfolio returns
    ret_bin = np.zeros([ret_full.shape[0],n_bins])
    
    # Construct portfolios
    for b in range(n_bins):
        # Bin portfolio
        bin_port = alpha_bin[alpha_bin['Bin']==b].copy()
        
        # Equal weight portfolio
        n_bin = bin_port.shape[0]
        bin_port['Weight'] = 1/n_bin
        
        # Stock weights
        bin_wt = bin_port['Weight']
        ret_bin[:,b] = np.dot(ret_full[bin_wt.index],bin_wt)
        
    # Calculate cumulative returns
    ret_bin_cum = pd.DataFrame(np.cumprod(ret_bin+1, axis = 0)-1, index = ret_full.index)
    
    return ret_bin_cum, alpha_bin

# Execution
if __name__ == '__main__':
    # Pull data
    px_close = eq.pull_hist('Close')
    t_end, n = px_close.shape
    
    # Calculate returns
    ret_full = pd.DataFrame(eq.calc_return(px_close), columns = px_close.columns, index = px_close.index[1:])
    
    # Lookback and holding periods
    lookback = 250
    holding = 400
    
    # Portfolio returns
    n_bins = 10
    ret_bin = pd.DataFrame(np.zeros([t_end-1,n_bins]), index = ret_full.index)
    
    # For every holding period
    for t in range(lookback,t_end,holding):
        # Cumulative return
        px_start = px_close.iloc[t-lookback,:]
        px_end = px_close.iloc[t,:]
        alpha = px_end/px_start-1
        
        # Remove nan
        alpha = alpha[~np.isnan(alpha)]    
        
        # Divide into bins
        alpha_bin = split_bin(alpha, n_bins)
        
        # Construct portfolios
        for b in range(n_bins):
            # Bin portfolio
            bin_stocks = alpha_bin[alpha_bin['Bin']==b]
            
            # Weights
            wts = px_close.iloc[0,:].copy()
            wts[:] = 0
            wts[bin_stocks.index] = 1/bin_stocks.shape[0]
            
            ret_bin.iloc[t:t+holding,b] = np.dot(ret_full.iloc[t:t+holding,:],wts)
            
    # Calculate equal weighted index of universe
    ret_bin['Index'] = ret_full.mean(axis = 1)
    # Calculate cumulative returns
    ret_bin_cum = pd.DataFrame(np.cumprod(ret_bin.iloc[lookback:,:]+1, axis = 0)-1, index = ret_full.index)
    
    # Plot
    ret_bin_cum.plot()