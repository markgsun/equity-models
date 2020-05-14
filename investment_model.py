# -*- coding: utf-8 -*-
"""
Created on Thu May 14 11:04:32 2020

@author: markg

Long-only investment model for medium horizons
"""

import numpy as np
import pandas as pd
import seaborn as sns
import equity_shared as eq

def split_bin(arr,n_bins):
    # Sort array
    arr_sort = pd.DataFrame(arr).sort_values(0)[10:-10]
    
    # Divide into buckets
    bin_labels = range(n_bins)
    arr_sort['Bin'] = pd.qcut(arr_sort.values.flatten(), n_bins, labels = bin_labels)
    
    return arr_sort

def alpha_mom_str(px_close, ti, t):
    # Cumulative return
    px_start = px_close.iloc[ti-t,:]
    px_end = px_close.iloc[ti,:]
    alpha = px_end/px_start-1
    
    # Remove nan
    alpha = alpha[~np.isnan(alpha)]    
    
    return alpha

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
    
    # Calculate alpha
    ti = 20
    t = 20
    alpha = alpha_mom_str(px_close, ti)  
    
    # Construct bin portfolios and calculate returns
    n_bins = 10
    ret_bin_cum, alpha_bin = bin_portfolios(px_close.iloc[ti:], alpha, n_bins)
    
    ret_bin_cum.plot()