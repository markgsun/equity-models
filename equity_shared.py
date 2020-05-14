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


# Calculate return
def calc_return(px):
    a = px.iloc[1:,:].sort_index().values
    b = px.iloc[0:-1,:].sort_index().values
    ret_full = (a/b)-1
    ret_full = np.nan_to_num(ret_full, nan = 0)
    
    return ret_full