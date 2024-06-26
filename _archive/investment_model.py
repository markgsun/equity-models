# -*- coding: utf-8 -*-
"""
Created on Thu May 14 11:04:32 2020

@author: markg

Long-only investment model for medium horizons
"""

import numpy as np
import pandas as pd
import equity_util as eq
import matplotlib.pyplot as plt
import seaborn as sns

def split_bin(arr,n_bins):
    # Add small random perturbations to array to remove duplicates
    arr = arr + np.random.normal(scale = 0.001, size = arr.shape)
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

def bin_construction(t, holding, px_close, px_vol, bk2mkt, alpha_wts, n_bins, ret_bin, ret_full):
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
        
        # Calculate returns
        ret_bin.iloc[t:t+holding,b] = np.dot(ret_full.iloc[t:t+holding,:],wts)
    
    return alpha_bin, ret_bin

def portfolio_backtest(px_close, px_vol, bk2mkt, sp_close, lookback, holding, n_bins, alpha_wts):
    t_end, n = px_close.shape
    
    # Calculate returns
    ret_full = pd.DataFrame(eq.calc_return(px_close), columns = px_close.columns, index = px_close.index[1:])
    
    # Construct empty portfolio returns
    ret_bin = pd.DataFrame(np.zeros([t_end-1,n_bins]), index = ret_full.index)
    
    # For every holding period
    for t in range(lookback,t_end,holding):
        # Portfolio construction
        alpha_bin, ret_bin = bin_construction(t, holding, px_close, px_vol, bk2mkt, alpha_wts, n_bins, ret_bin, ret_full)
    
    # Calculate equal weighted index of universe
    ret_bin['Average'] = ret_full.mean(axis = 1)
    
    # Calculate S&P500 returns
    sp_ret = pd.DataFrame(eq.calc_return(sp_close), columns = sp_close.columns, index = sp_close.index[1:])
    ret_bin = ret_bin.join(sp_ret)
    
    # Calculate cumulative returns
    ret_bin_cum = pd.DataFrame(np.cumprod(ret_bin.iloc[lookback:,:]+1, axis = 0)-1, index = ret_full.index)
    
    # Update column names
    cols = ['First Bin']+['Mid Bins']*(n_bins-2)+['Last Bin']+['Average']+['Index']
    ret_bin_cum.columns = cols
    
    return ret_bin_cum, alpha_bin

def visualize_backtest(ret_bin_cum, alpha_wts):
    # Melt data for plot
    ret_long = pd.melt(ret_bin_cum.reset_index(), id_vars = 'index')
    ret_long.columns = ['Date','Bin','Return']
    
    # Plot
    fig, ax = plt.subplots(figsize = (12,5), dpi = 100)
    sns.lineplot('Date', 'Return', hue = 'Bin', data=ret_long, ax = ax, legend = 'brief')
    plt.title('Binned portfolio cumulative returns '+str(alpha_wts))

def long_short_investments(start, end, idx, lookback, holding, n_bins, alpha_wts):
    # Pull data
    px_close = eq.pull_hist('Close', start, end, idx = idx)
    px_vol = eq.pull_hist('Volume', start, end, idx = idx)
    bk2mkt = eq.pull_bk2mkt(start, end, px_close, idx = idx)
    sp_close = eq.pull_hist_ind('Close',start, end)

    # Compute portfolio returns
    ret_bin_cum, alpha_bin = portfolio_backtest(px_close, px_vol, bk2mkt, sp_close, lookback, holding, n_bins, alpha_wts)
    
    # Plot
    visualize_backtest(ret_bin_cum, alpha_wts)
    
    # Identify long and short investments at most recent time period
    longs = alpha_bin[alpha_bin['Bin']==0]
    shorts = alpha_bin[alpha_bin['Bin']==n_bins-1]
    
    return longs, shorts, ret_bin_cum, alpha_bin
    

# Execution
if __name__ == '__main__':
    # Parameters
    start = '2017-01-01'
    end = '2020-07-01'
    idx = 'S&P 500'
    lookback = 250
    holding = 60
    n_bins = 50
    alpha_wts = wts = {'Momentum':1,'Volume':1,'Book2Market':1}
    
    # Long-short model
    longs, shorts, ret_bin_cum, alpha_bin = long_short_investments(start, end, idx, lookback, holding, n_bins, alpha_wts)