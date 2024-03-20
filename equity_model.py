# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18

@author: Mark Sun

Quantitative Equity Model built upon scraped financials
"""

# Packages
import numpy as np
import pandas as pd
from datetime import datetime as dt
from bokeh.plotting import figure, show, output_file

# Personal modules
import equity_util as eq


class EquityModel:
    def __init__(self, idx, start, end):
        # Pull stock and index prices
        self.stk_px = eq.pull_hist('Close', start, end, idx)
        self.idx_px = eq.pull_hist_ind('Close', start, end, idx)
        self.stk_dt = self.stk_px.index

        # Compute daily returns
        self.stk_ret = eq.calc_return(self.stk_px)
        self.idx_ret = eq.calc_return(self.idx_px)

        # Additional attributes
        self.stk_fd = self.pull_fundamentals(start, end, self.stk_px.columns)

    # Pull fundamental data
    def pull_fundamentals(self, start, end, stocks):
        fd_list = ['Basic Average Shares', "Total stockholders' equity"]
        stk_dt = list(self.stk_dt)
        stk_fd = {}
        for fd in fd_list:
            # Pull stock fundamentals
            stk_fd_tmp = pd.DataFrame(index=self.stk_dt, columns=self.stk_px.columns)
            stk_fd_res = eq.pull_fd(fd, start, end, stocks)

            # Align dates for fundamentals with those for prices
            for d, date in enumerate(sorted(list(stk_fd_res.index))):
                if date not in stk_dt:
                    dff = [x.days for x in np.asarray(stk_dt) - date]
                    try:
                        ind = dff.index(min([x for x in dff if x > 0]))
                    except ValueError:
                        continue
                    fill_dt = stk_dt[ind]
                else:
                    fill_dt = date
                stk_fd_tmp.loc[fill_dt, :] = stk_fd_res.loc[date, :]
            stk_fd[fd] = stk_fd_tmp.sort_index().astype(float).ffill()

        return stk_fd

    # Calculate book to market signal
    def alpha_b2m(self):
        mrkt = self.stk_fd['Basic Average Shares']*self.stk_px
        book = self.stk_fd["Total stockholders' equity"]
        b2mk = (book/mrkt).replace([np.inf, -np.inf], np.nan)
        for date in self.stk_dt:
            b2mk.loc[date, :] = eq.windsorize(b2mk.loc[date, :])
        # print(b2mk[~b2mk.isnull().all(axis=1)])
        return b2mk

    # Calculate size signal
    def alpha_size(self):
        mrkt = self.stk_fd['Basic Average Shares']*self.stk_px
        for date in self.stk_dt:
            mrkt.loc[date, :] = -eq.windsorize(mrkt.loc[date, :])
        return mrkt

    # Calculate momentum signal
    def alpha_mom(self, k=250):
        lag1 = self.stk_px.shift(1)
        lag2 = self.stk_px.shift(k)
        momk = lag1/lag2-1
        for date in self.stk_dt:
            momk.loc[date, :] = eq.windsorize(momk.loc[date, :])
        return momk

    # Calculate mean reversion signal
    def alpha_rev(self, k=5):
        lag1 = self.stk_px.shift(1)
        lag2 = self.stk_px.shift(k)
        revk = (lag1/lag2-1)
        for date in self.stk_dt:
            revk.loc[date, :] = -eq.windsorize(revk.loc[date, :])
        return revk

    # Construct portfolio weights
    def port_wt_ls(self, alpha_wt, q=0.1):
        # Compute alphas
        a_b2m = self.alpha_b2m()
        a_size = self.alpha_size()
        a_mom = self.alpha_mom()
        a_rev = self.alpha_rev()

        # Get relative weights
        wt_tot = sum([x for x in alpha_wt.values()])
        alpha_wt = {x: alpha_wt[x]/wt_tot for x in alpha_wt.keys()}

        # Weigh alpha
        alpha = (a_b2m*alpha_wt['b2m'] +
                 a_size*alpha_wt['size'] +
                 a_mom*alpha_wt['mom'] +
                 a_rev*alpha_wt['rev'])

        # Sort alphas by quantiles
        q_lo = alpha.quantile(q=q, axis=1)
        q_hi = alpha.quantile(q=1-q, axis=1)

        # Identify lowest and top quantiles
        bool_lo = alpha.lt(q_lo, axis=0)
        bool_hi = alpha.gt(q_hi, axis=0)

        # Weight stocks by quantiles
        stk_wt = pd.DataFrame(index=self.stk_dt, columns=self.stk_px.columns)
        for date in self.stk_dt:
            # Shorts
            stk_lo = list(alpha.loc[date, bool_lo.loc[date, :]].index)
            if len(stk_lo):
                stk_wt.loc[date, stk_lo] = -1/len(stk_lo)
            # Longs
            stk_hi = list(alpha.loc[date, bool_hi.loc[date, :]].index)
            if len(stk_hi):
                stk_wt.loc[date, stk_hi] = 1/len(stk_hi)
        stk_wt = stk_wt.astype(float).fillna(0)

        return stk_wt

    # Backtest strategies
    def backtest(self, start, end, alpha_wt, icap, desc, rebal=20):
        # Get target weights
        wts_all = self.port_wt_ls(alpha_wt)
        tgt_wts = wts_all.loc[(wts_all.index >= dt.strptime(start, '%Y-%m-%d').date()) &
                              (wts_all.index <= dt.strptime(end, '%Y-%m-%d').date()), :]
        bk_dts = tgt_wts.index

        # Backtest
        stk_pos = pd.DataFrame(0.0, index=bk_dts, columns=tgt_wts.columns)
        stk_pnl = pd.DataFrame(0.0, index=bk_dts, columns=tgt_wts.columns)

        # Initialize portfolio
        stk_pos.iloc[0, :] = tgt_wts.iloc[0, :] * icap
        # print(tgt_wts.iloc[0, :][tgt_wts.iloc[0, :] > 0])
        # test = 'ABBV'
        # print(f'Position: {stk_pos.loc[bk_dts[0], test]}')
        for d, date in enumerate(bk_dts):
            if not d:
                continue

            # PnL by stock
            stk_pnl.loc[date, :] = stk_pos.iloc[d-1, :] * self.stk_ret.loc[date, :]
            stk_pos.loc[date, :] = (self.stk_ret.loc[date, :] + 1) * stk_pos.iloc[d-1, :]
            # print(f'\nDate: {date}')
            # print(f'Price: {self.stk_px.loc[date, test]}')
            # print(f'Ret: {self.stk_ret.loc[date, test]}')
            # print(f'Position: {stk_pos.loc[date, test]}')
            # print(f'P/L: {stk_pnl.loc[date, test]}')

            # Rebalance
            if not d % rebal:
                # print('Rebal')
                # gross_val = (sum(stk_pos.loc[date, stk_pos.loc[date, :] > 0.0]) -
                #              sum(stk_pos.loc[date, stk_pos.loc[date, :] < 0.0]))
                gross_val = icap
                stk_pos.loc[date, :] = tgt_wts.loc[date, :] * gross_val
                # print(f'Position: {stk_pos.loc[date, test]}')

        # Portfolio level pnl
        port_pnl = stk_pnl.sum(axis=1)
        port_pnl_cum = port_pnl.cumsum()

        # Visualize pnl
        output_file(filename=f'{desc}.html')
        p = figure(width=1000, height=400, title=f'Portfolio Backtest - Cumulative PnL - {desc}',
                   x_axis_label='Date', y_axis_label='PnL', x_axis_type='datetime')
        p.line(x=port_pnl_cum.index, y=port_pnl_cum.values)
        show(p)

        return None


# Execution
if __name__ == '__main__':
    data_start = '2021-12-31'
    data_end = '2023-12-31'
    test_start = '2022-12-31'
    test_end = '2023-12-31'
    indx = 'S&P 500'
    init_cap = 100000
    alpha_weights = {'b2m': 0, 'size': 0, 'mom': 0, 'rev': 1}
    description = '_Mean Reversion Factor'

    mdl = EquityModel(indx, data_start, data_end)
    mdl.backtest(test_start, test_end, alpha_weights, init_cap, description)
