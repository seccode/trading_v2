'''
This is the most beautiful function that has ever been written
'''
import datetime
import pandas as pd

date_format_1 = '%Y-%m-%dT%H:%M:%S.000000000Z'  # Format in data file
date_format_2 = '%m/%d/%Y-%H:%M:%S'             # Format entered by user

def data_loader(currency='EUR_USD',start='12/15/2019-02:09:15',end='12/15/2019-02:09:15',granularity='S5'):
    # Do not pick days/times that are when market is closed
    filename = 'data/'+granularity+'_clean_'+currency+'.csv'

    # Load file into dataframe
    df = pd.read_csv(filename,header=None,index_col=0,
                    names=['datetime','time','bid_open','bid_high','bid_low','bid_close','ask_open','ask_high','ask_low','ask_close','volume'])

    # Slice data according to start and end date
    data_mat = df.loc[datetime.datetime.strptime(start,date_format_2).strftime(date_format_1):datetime.datetime.strptime(end,date_format_2).strftime(date_format_1),
                    ['time','bid_high','bid_low','bid_close','ask_high','ask_low','ask_close','volume']].values

    return data_mat
