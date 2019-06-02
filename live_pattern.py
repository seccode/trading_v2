'''
Script for live trading of pattern strategy
'''

import datetime
import pandas as pd
from data_loader import data_loader
from oanda_interface import oanda_interface
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from scipy.stats import levene
import numpy as np
import time


def reverse_price_scale(a1):
    vals = [0]
    for x in range(1,len(a1)):
        vals.append(np.mean(a1[len(a1)-1-x]) - np.mean(a1[len(a1)-1]))
    return np.array(vals[::-1])

def price_scale(a1):
    vals = [0]
    for x in range(1,len(a1)):
        vals.append(np.mean(a1[x]) - np.mean(a1[0]))
    return np.array(vals)

class Trader():
    def __init__(self,currency,train_data):
        self.currency = currency
        self.train_data = train_data
        self.open_positions = {}

    def add_time(self,time):
        '''Function to add look forward time to given time'''
        hours = int(time/100)
        minutes = (time % 100) + self.look_forward
        hours_to_add = int(minutes/60)
        minutes_to_add = minutes - 60*hours_to_add
        end_time = (hours*100) + (100*hours_to_add) + minutes_to_add
        return end_time

    def get_current_time(self):
        now = datetime.datetime.now()
        return now.hour*100 + now.minute

    def make_patterns(self):
        # Loop through training data and create stored patterns
        patterns = []
        outcomes = []
        volumes = []
        i = self.look_back
        while i < self.train_data.shape[0] - self.look_forward:
            patterns.append(reverse_price_scale(self.train_data[i-self.look_back:i+1,3]))
            outcomes.append(price_scale(self.train_data[i:i+self.look_forward,3]))
            volumes.append(self.train_data[i-self.look_back:i+1,7])
            i += 1
        self.patterns = np.array(patterns)
        self.outcomes = np.array(outcomes)
        self.volumes = np.array(volumes)

    def long_trade(self,TAKE,STOP,POSITION_SIZE):
        print("Long")
        # 1. Record prices
        price_info = tradeInterface.get_instrument_info(self.currency,'M1',1)
        buy_price = price_info['ask_close'][0]
        current_acct_value = float(tradeInterface.get_account_information()['account']['balance'])
        print("Current Account Value: {}".format(current_acct_value))

        # 2. Calculate Stop and Take
        take_price = round(buy_price + TAKE,4)
        # 3. Round price requests as necessary
        if self.currency == 'USD_JPY':
            take_price = round(take_price,3)
            STOP = round(STOP,3)
        # 4. Open position
        trade_ID = tradeInterface.open_position(self.currency,POSITION_SIZE,'long',take_profit=take_price,trail_stop=STOP)
        if trade_ID == None:
            trade_ID = tradeInterface.open_position(self.currency,POSITION_SIZE,'long',take_profit=take_price,trail_stop=STOP)

        # 5. Update open positions
        if trade_ID == None:
            return
        self.open_positions[trade_ID] = {'type':'Long','enter_price':buy_price,'enter_time':self.get_current_time()}


    def short_trade(self,TAKE,STOP,POSITION_SIZE):
        print("Short")
        # 1. Record prices
        price_info = tradeInterface.get_instrument_info(self.currency,'M1',1)
        sell_price = price_info['bid_close'][0]
        current_acct_value = float(tradeInterface.get_account_information()['account']['balance'])
        print("Current Account Value: {}".format(current_acct_value))

        # 2. Calculate Stop and Take
        take_price = round(sell_price - TAKE,4)
        # 3. Round price requests as necessary
        if self.currency == 'USD_JPY':
            take_price = round(take_price,3)
            STOP = round(STOP,3)
        # 4. Open position
        trade_ID = tradeInterface.open_position(self.currency,POSITION_SIZE,'short',take_profit=take_price,trail_stop=STOP)
        if trade_ID == None:
            trade_ID = tradeInterface.open_position(self.currency,POSITION_SIZE,'short',take_profit=take_price,trail_stop=STOP)

        # 5. Update open positions
        if trade_ID == None:
            return
        self.open_positions[trade_ID] = {'type':'Short','enter_price':sell_price,'enter_time':self.get_current_time()}


    def trade(self,parameters):
        self.look_back = int(parameters[0])
        self.look_forward = int(parameters[1])
        self.thresh = round(parameters[2],5)
        self.vol_thresh = round(parameters[3],2)
        self.match_num = int(parameters[4])

        self.make_patterns()

        self.index = 0
        while True:
            print(self.index)
            for key, value in self.open_positions.items():
                if self.get_current_time() > add_time(value['enter_time']):
                    tradeInterface.close_position(key)
                    del self.open_positions[key]

            # Get current pattern
            price_info = tradeInterface.get_instrument_info(self.currency,'M1',self.look_back+1)
            spread = price_info['ask_close'][-1] - price_info['bid_close'][-1]
            current_pattern = reverse_price_scale(price_info['bid_close'])
            current_volume = price_info['volume']

            cs = euclidean_distances(self.patterns,current_pattern.reshape(1,-1))

            inds = np.where(cs < np.sort(cs.flatten())[40])[0]
            if inds.shape[0] < 4:
                self.index += 1
                time.sleep(WAIT_TIME_SECONDS)
                continue

            new_inds = []
            j = 0
            while j < inds.shape[0]:
                if ((j == 0) or (j != 0 and inds[j] - inds[j-1] > self.look_back/2)) and \
                    levene(current_volume,self.volumes[inds[j]])[1] > self.vol_thresh:

                    new_inds.append(inds[j])
                j += 1
            inds = np.array(new_inds)

            if inds.shape[0] > self.match_num and np.abs(np.median(self.outcomes[inds,-1])) > self.thresh:
                plt.plot(current_pattern,'k')
                for ind in inds:
                    p = plt.plot(self.patterns[ind],alpha=.4)
                    plt.plot(range(self.look_back,self.look_back+self.look_forward),self.outcomes[ind],alpha=.4,c=p[0].get_color())
                plt.show()

                if np.median(self.outcomes[inds,-1]) > 0:
                    self.long_trade(np.median(self.outcomes[inds,-1]),-np.min(self.outcomes[inds])+2*spread,FIXED_POSITION_SIZE)
                else:
                    self.short_trade(-np.median(self.outcomes[inds,-1]),np.max(self.outcomes[inds])+2*spread,FIXED_POSITION_SIZE)

            self.index += 1
            time.sleep(WAIT_TIME_SECONDS)







# Initialize API
'''Sam's Live Account'''
# FIXED_POSITION_SIZE = 10000
# tradeInterface = oanda_interface('001-001-2154685-001','83bfe1b504bd65b07513a811b630993f-cbf32b5d40d2bc7827f11308d2a5b55c')
'''Brendan's Live Account'''
# FIXED_POSITION_SIZE = 20000
# tradeInterface = oanda_interface('001-001-2154685-001','83bfe1b504bd65b07513a811b630993f-cbf32b5d40d2bc7827f11308d2a5b55c')
'''Sam's Demo Account'''
FIXED_POSITION_SIZE = 60000
tradeInterface = oanda_interface('101-001-8846728-001','bce4f5138dfcd1a12b028693c23ada2d-71e9ffa5dd3c1109730bc6c92d8cf9a7',environment='practice')
'''Brendan's Demo Account'''
# FIXED_POSITION_SIZE = 60000
# tradeInterface = oanda_interface('101-001-8734770-001','badcf7760c1558e67da9dcc7144be117-e4dc07376ebb83f374fb3a7cb82725f6',environment='practice')

print(tradeInterface.get_instrument_info('EUR_USD','M1',1))
# info = tradeInterface.get_calendar_info('EUR_USD',1286400)
# for item in info:
    # print(item)
    # print(item['title'],datetime.datetime.fromtimestamp(item['timestamp']).strftime('%Y-%m-%d %H:%M:%S'))

RISK_VALUE = .01
WAIT_TIME_SECONDS = 2

# currency = 'EUR_USD'
# parameters = [116, 34, 0.00059, 0.04, 5]

# train_data, _ = data_loader('EUR_USD','02/26/19','2100','02/25/19','2100')

# m = Trader(currency,train_data)
# m.trade(parameters)








#
