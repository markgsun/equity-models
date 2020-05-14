# -*- coding: utf-8 -*-
"""
Created on Thu May 14 11:04:32 2020

@author: markg

Long-only investment model for medium horizons
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

# Execution
if __name__ == '__main__':
    # Pull data
    px_close = pull_hist('Close')
    
    # Starting time 
    ti = 250
    
    # Cumulative return
    ret = data_period.iloc[-1,:]/data_period.iloc[0,:]-1