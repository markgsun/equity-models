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
from urllib import parse

# Establish connection to database
def sqlconn(db):
    params = parse.quote_plus('DRIVER={SQL Server};'
                              'SERVER=DESKTOP-9HGRDTD\SQLEXPRESS;'
                              'DATABASE='+db+';'
                              'Trusted_Connection=yes')

    engine = create_engine('mssql+pyodbc:///?odbc_connect={}'.format(params))
    return engine

# Pull historical price data
def pull_hist(val, start, end, idx = ''):
    # Connect to database
    engine = sqlconn('FinancialData')
    
    # Execute SQL statement
    statement = '''
                SELECT E.*, S."Index" FROM EquityPx E
                INNER JOIN SecurityMaster S
                ON E.Stock = S.Stock
                WHERE S."Index" LIKE CONCAT('{}','%')
                AND CONVERT(datetime, E.Date) BETWEEN '{}' AND '{}';
                '''.format(idx, start, end)
    res = engine.execute(statement)
    data_raw = pd.DataFrame(res.fetchall())
    data_raw.columns = res.keys()
    
    # Get closing price
    px_long = data_raw[[val,'Stock','Date']]
    
    # Reshape price data
    px_wide = px_long.pivot_table(values = val,index = 'Date',columns = 'Stock', aggfunc = 'mean')
    
    # Convert index to dates
    px_wide.index = [dt.datetime.strptime(date, '%b %d, %Y').date() for date in px_wide.index.values]
    
    # Sort by date
    px_wide = px_wide.sort_index()
    
    return px_wide

# Pull book to market data
def pull_bk2mkt(start, end, px_close, idx = ''):
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
    bk2mkt_long = data_raw[['bk2mkt','Stock','Date']].copy()
    
    # Convert index to dates
    bk2mkt_long['Date'] = [dt.datetime.strptime(date, '%m/%d/%Y').date() for date in bk2mkt_long['Date']]
    
    # Get wide shape based on historical prices
    bk2mkt_wide = px_close.copy()
    bk2mkt_wide.iloc[:,:] = float('nan')
    
    # Fill in wide dataframe
    for i in bk2mkt_long.index:
        try:
            stock_tmp = bk2mkt_long.loc[i,'Stock']
            date_tmp = bk2mkt_long.loc[i,'Date']
            bk2mkt_wide.loc[date_tmp,stock_tmp]
            bk2mkt_wide.loc[date_tmp,stock_tmp] = bk2mkt_long.loc[i,'bk2mkt']
        except:
            pass
    # Sort by date
    bk2mkt_wide = bk2mkt_wide.sort_index()
    
    # Fill nans
    bk2mkt_wide = bk2mkt_wide.bfill(axis = 'rows').ffill(axis = 'rows').fillna(0)
    
    return bk2mkt_wide
    

# Calculate return
def calc_return(px):
    a = px.iloc[1:,:].sort_index().values
    b = px.iloc[0:-1,:].sort_index().values
    ret_full = (a/b)-1
    ret_full = np.nan_to_num(ret_full, nan = 0)
    
    return ret_full

# Execution
if __name__ == '__main__':
    
    start = '2017-01-01'
    end = '2020-05-01'
    idx = 'S&P 500'
    px_close = pull_hist('Close', start, end, idx = idx)
    
    bk2mkt_wide = pull_bk2mkt(start, end, px_close, idx = idx)