# -*- coding: utf-8 -*-
"""
Created on Thu May 14 11:30:34 2020

@author: markg

Shared equity model functions
"""

import datetime as dt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import text
from urllib import parse


# Establish connection to database
def sqlconn():
    engine = create_engine('mysql+mysqlconnector://fduser:%s@localhost/financialdata'
                           % parse.quote_plus('#dy@2PcGKdACyJ'))
    return engine


# Pull historical price data
def pull_hist(val, start, end, idx):
    # Connect to database
    engine = sqlconn()

    # Execute SQL statement
    statement = f'''
        SELECT 
            STR_TO_DATE(E.date, '%M %d, %Y') as Date,
            E.Stock,
            E.{val},
            S.`Index` as Indx
        FROM EquityPx E
        INNER JOIN SecurityMaster S
        ON E.Stock = S.Stock
        WHERE S.`Index` = "{idx}"
        AND STR_TO_DATE(E.date, '%M %d, %Y')
            BETWEEN '{start}' AND '{end}'
    '''
    with engine.begin() as conn:
        res = conn.execute(text(statement))

    data_raw = pd.DataFrame(res.fetchall())
    data_raw.columns = res.keys()

    # Get closing price
    px_long = data_raw[[val, 'Stock', 'Date']]

    # Reshape price data
    px_wide = px_long.pivot_table(values=val, index='Date', columns='Stock', aggfunc='mean')

    # Sort by date
    px_wide = px_wide.sort_index()

    return px_wide


# Pull historical index data
def pull_hist_ind(val, start, end, idx_raw='S&P 500'):
    idx = {'S&P 500': '%5EGSPC'}[idx_raw]

    # Connect to database
    engine = sqlconn()

    # Execute SQL statement
    statement = f'''
            SELECT 
                STR_TO_DATE(Date, '%M %d, %Y') as Date,
                Stock,
                {val}
            FROM IndexPx
            WHERE Stock = "{idx}"
            AND STR_TO_DATE(date, '%M %d, %Y')
                BETWEEN '{start}' AND '{end}'
        '''
    with engine.begin() as conn:
        res = conn.execute(text(statement))

    data_raw = pd.DataFrame(res.fetchall())
    data_raw.columns = res.keys()

    # Get closing price
    px_long = data_raw[[val, 'Stock', 'Date']]

    # Reshape price data
    px_wide = px_long.pivot_table(values=val, index='Date', columns='Stock', aggfunc='mean')

    # Sort by date
    px_wide = px_wide.sort_index()

    return px_wide


# Pull fundamentals
def pull_fd(fd, start, end, stocks):
    # Connect to database
    engine = sqlconn()

    # Execute SQL statement
    statement = f'''
        SELECT 
            STR_TO_DATE(Date, '%M %d, %Y') as Date,
            Stock,
            Field,
            Value
        FROM EquityFd
        WHERE Field = "{fd}"
            AND STR_TO_DATE(Date, '%M %d, %Y')
            BETWEEN '{start}' AND '{end}'
    '''
    with engine.begin() as conn:
        res = conn.execute(text(statement))

    data_raw = pd.DataFrame(res.fetchall())
    data_raw.columns = res.keys()

    # Filter on stock list
    fd_long = data_raw.loc[data_raw['Stock'].isin(stocks), :]

    # Reshape data
    fd_wide = fd_long.pivot_table(values='Value', index='Date', columns='Stock')
    fd_wide = fd_wide.sort_index().ffill()

    return fd_wide


# Pull book to market data
def pull_bk2mkt(start, end, px_close, idx=''):
    # Connect to database
    engine = sqlconn('FinancialData')

    # Execute SQL statement
    statement = '''
                SELECT B.*, S."Index" FROM Book2Market B
                INNER JOIN SecurityMaster S
                ON B.Stock = S.Stock
                WHERE S."Index" LIKE CONCAT('{}','%')
                AND CONVERT(datetime, B.Date) BETWEEN '{}' AND '{}';
                '''.format(idx, start, end)
    res = engine.execute(statement)
    data_raw = pd.DataFrame(res.fetchall())
    data_raw.columns = res.keys()

    # Get ratio
    bk2mkt_long = data_raw[['bk2mkt', 'Stock', 'Date']].copy()

    # Convert index to dates
    bk2mkt_long['Date'] = [dt.datetime.strptime(date, '%m/%d/%Y').date() for date in bk2mkt_long['Date']]

    # Get wide shape based on historical prices
    bk2mkt_wide = px_close.copy()
    bk2mkt_wide.iloc[:, :] = float('nan')

    # Fill in wide dataframe
    for i in bk2mkt_long.index:
        try:
            stock_tmp = bk2mkt_long.loc[i, 'Stock']
            date_tmp = bk2mkt_long.loc[i, 'Date']
            bk2mkt_wide.loc[date_tmp, stock_tmp] = bk2mkt_long.loc[i, 'bk2mkt']
        except Exception as err:
            print(err)
            pass
    # Sort by date
    bk2mkt_wide = bk2mkt_wide.sort_index()

    # Fill nans
    bk2mkt_wide = bk2mkt_wide.bfill(axis='rows').ffill(axis='rows').fillna(0)

    return bk2mkt_wide


# Calculate return
def calc_return(px):
    a = px.iloc[1:, :].sort_index().values
    b = px.iloc[0:-1, :].sort_index().values
    ret_full = (a / b) - 1
    ret_full = np.nan_to_num(ret_full, nan=0)

    ret_pd = pd.DataFrame(data=ret_full,
                          index=px.index[1:],
                          columns=px.columns)
    return ret_pd


def windsorize(raw, n=3):
    ct = 0
    out = raw
    while ct < n:
        # Standardize
        out = (out - out.mean()) / out.std()
        # Windsorize
        out[abs(out) > 3] = 3 * out[abs(out) > 3] / abs(out[abs(out) > 3])
        ct += 1

    return out


# Straight momentum alpha
def alpha_mom_str(data_period, windsor=True):
    # Cumulative return
    a_mom = data_period.iloc[0, :] / data_period.iloc[-1, :] - 1
    # Windsorize
    if windsor:
        a_mom = windsorize(a_mom, 10)

    return a_mom


# Weekly volume alpha
def alpha_vol_wk(data_period, windsor=True):
    data_1wk = data_period.iloc[-5:-1, :]

    # Average volume
    a_vol = data_1wk.mean(axis=0)

    # Windsorize
    if windsor:
        a_vol = windsorize(a_vol, 10)

    return a_vol


# Book to market alpha
def alpha_b2m(data_period, windsor=True):
    # Book to Market
    a_b2m = -data_period

    # Windsorize
    if windsor:
        a_b2m = windsorize(-data_period, 10)

    return a_b2m


# Alpha model
def alpha_model(t, px_close, px_vol, bk2mkt, windsor=True, wts=None):
    # Calculate individual alphas
    if wts is None:
        wts = {'Momentum': 1, 'Volume': 1, 'Book2Market': 1}

    a_mom = alpha_mom_str(px_close.iloc[t - 250:t, :], windsor)
    a_vol = alpha_vol_wk(px_vol.iloc[t - 5:t, :], windsor)
    a_b2m = alpha_b2m(bk2mkt.iloc[t, :], windsor)

    # Pull numeric weights
    wts_num = [wts['Momentum'], wts['Volume'], wts['Book2Market']]

    # Aggregate alphas
    alphas = pd.concat([a_mom, a_vol, a_b2m], axis=1) * wts_num / sum(wts_num)
    alpha = alphas.sum(axis=1)

    return alpha


# Execution
if __name__ == '__main__':
    pass
