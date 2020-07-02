# -*- coding: utf-8 -*-
"""
Created on Thu May 14 11:04:32 2020

@author: markg

Long-only investment model for medium horizons
"""

import numpy as np
import pandas as pd
import equity_shared as eq
import matplotlib.pyplot as plt
import seaborn as sns

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

def portfolio_backtest(px_close, px_vol, bk2mkt, lookback, holding, n_bins, alpha_wts):
    t_end, n = px_close.shape
    
    # Calculate returns
    ret_full = pd.DataFrame(eq.calc_return(px_close), columns = px_close.columns, index = px_close.index[1:])
    
    # Construct empty portfolio returns
    ret_bin = pd.DataFrame(np.zeros([t_end-1,n_bins]), index = ret_full.index)
    
    # For every holding period
    for t in range(lookback,t_end,holding):
        # Alpha
        alpha = eq.alpha_model(t, px_close, px_vol, bk2mkt, windsor = True, wts = alpha_wts)  
        
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
    ret_bin['Average'] = ret_full.mean(axis = 1)
    # Calculate cumulative returns
    ret_bin_cum = pd.DataFrame(np.cumprod(ret_bin.iloc[lookback:,:]+1, axis = 0)-1, index = ret_full.index)

    # Update column names
    cols = ['First Bin']+['Mid Bins']*8+['Last Bin']+['Average']
    ret_bin_cum.columns = cols
    
    return ret_bin_cum

def vizualize_backtest(ret_bin_cum):
    # Melt data for plot
    ret_long = pd.melt(ret_bin_cum.reset_index(), id_vars = 'index')
    ret_long.columns = ['Date','Bin','Return']
    
    # Plot
    fig, ax = plt.subplots(figsize = (12,5), dpi = 100)
    sns.lineplot('Date', 'Return', hue = 'Bin', data=ret_long, ax = ax, legend = 'brief')
    plt.title('Binned portfolio cumulative returns')

# Execution
if __name__ == '__main__':
    # Parameters
    start = '2017-01-01' 
    end = '2020-05-01'
    idx = 'S&P 500'
    lookback = 250
    holding = 60
    n_bins = 10
    alpha_wts = [1,1,1]
    
    # Pull data
    px_close = eq.pull_hist('Close', start, end, idx = idx)
    px_vol = eq.pull_hist('Volume', start, end, idx = idx)
    bk2mkt = eq.pull_bk2mkt(start, end, px_close, idx = idx)
    
    # Compute portfolio returns
    ret_bin_cum = portfolio_backtest(px_close, px_vol, bk2mkt, lookback, holding, n_bins, alpha_wts)

    # Plot
    vizualize_backtest(ret_bin_cum)
