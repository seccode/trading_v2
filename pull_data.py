'''
Script to get 5 second data from oanda and fill in gaps
'''

import csv
import datetime
import matplotlib.pyplot as plt
from oanda_interface import oanda_interface
import numpy as np
import time

tradeInterface = oanda_interface('001-001-2154685-001','83bfe1b504bd65b07513a811b630993f-cbf32b5d40d2bc7827f11308d2a5b55c')
date_format = '%Y-%m-%dT%H:%M:%S.000000000Z'

def getPriceInfo(from_time,instrument='EUR_USD',numPasts=5000,dataSet=False,gran='S5'):
    while True:
        try:
            prices_dict = tradeInterface.get_instrument_info(instrument,gran,numPasts,start_time=from_time)

            return candleToDataSet(prices_dict)
        except Exception as e:
            print(e)
            hour,minute = get_current_time()
            print(hour*100 + minute)
            print('Getting price info failed, trying again.')

def get_current_time():
    now = datetime.datetime.now()
    return now.hour, now.minute

def candleToDataSet(dict):
    total_return_mat = [dict['time'],
                        dict['bid_open'],
                        dict['bid_high'],
                        dict['bid_low'],
                        dict['bid_close'],
                        dict['ask_open'],
                        dict['ask_high'],
                        dict['ask_low'],
                        dict['ask_close'],
                        dict['volume']]

    return np.array(total_return_mat)


def get_data(granularity='S5'):

    start_time = '2018-01-03T00:00:00.000000000Z'

    not_reached_end = True
    num = 5000
    while not_reached_end:
        existingFile = open('data/'+granularity+'_raw_EUR_USD.csv','a')
        file_writer = csv.writer(existingFile)
        data = getPriceInfo(start_time,instrument='EUR_USD',numPasts=num,gran=granularity)

        for x in range(len(data[0])):
            if x == 0:
                continue
            file_writer.writerow([data[0][x],
                                data[1][x],
                                data[2][x],
                                data[3][x],
                                data[4][x],
                                data[5][x],
                                data[6][x],
                                data[7][x],
                                data[8][x],
                                data[9][x]])
        existingFile.close()
        start_time = data[0][-1]
        print(start_time)
        year = start_time.split('-')[0]
        month = start_time.split('-')[1]
        day = start_time.split('-')[2].split('T')[0]
        hour = start_time[11:13]
        minute = start_time[14:15]
        if year == '2019':
            if month == '05':
                if day == '31':
                    not_reached_end = False

def check_time(time):
    '''Check if time is when market is closed, returns False if market is closed'''
    date_info = str(time).split(' ')[0].split('-')
    time_info = str(time).split(' ')[1].split(':')
    current_weekday = datetime.datetime(int(date_info[0]),int(date_info[1]),int(date_info[2])).weekday()
    if current_weekday == 6:
        return False
    elif current_weekday == 5:
        if int(time_info[0]) >= 21:
            return False
    elif current_weekday == 0:
        if int(time_info[0]) < 21:
            return False
    return True

def edit_row(row):
    '''Add stand-alone time element to data row'''
    time_info = str(datetime.datetime.strptime(row[0],date_format)).split(' ')[1].split(':')
    return np.concatenate(([row[0]],[int(time_info[0])*100+int(time_info[1])+int(time_info[2])/100],row[1:]))

def clean_data(granularity='S5'):
    if granularity == 'S5':
        num = 5
    else:
        num = 60
    new_mat = []
    newFile = open('data/'+granularity+'_clean_EUR_USD.csv','w')
    existingFile = open('data/'+granularity+'_raw_EUR_USD.csv','r')
    csv_reader = csv.reader(existingFile)
    csv_writer = csv.writer(newFile)
    mat = np.array(list(csv_reader))
    for i, row in enumerate(mat):
        if i == mat.shape[0]-1:
            break
        current_date = datetime.datetime.strptime(row[0],date_format)
        forward_date = datetime.datetime.strptime(mat[i+1][0],date_format)
        delta = forward_date - current_date
        # Find instances where gap is greater than 5 seconds and fill gap

        new_mat.append(edit_row(row))
        if delta.seconds > num:
            for x in np.linspace(0,delta.seconds,int(delta.seconds/num+1)):
                if x == 0 or x == delta.seconds:
                    continue
                add_time = datetime.timedelta(seconds=x)
                fill_date = current_date + add_time
                # Check if time is when market is closed
                if check_time(fill_date):
                    new_row = mat[i].copy()
                    new_row[0] = fill_date.strftime(date_format)
                    new_mat.append(edit_row(new_row))
                else:
                    break
    for row in new_mat:
        csv_writer.writerow(row)

get_data(granularity='M1')
clean_data(granularity='M1')






















#
