'''
Pattern Strategy
'''

from scipy.stats import levene
from scipy.stats import pearsonr
from bayesianOpt import bayesian_optimisation
from sklearn.metrics import mean_squared_error
from data_loader import data_loader
import matplotlib.pyplot as plt
import numpy as np


def price_scale(a1):
    vals = [0]
    for x in range(1,len(a1)):
        vals.append(a1[len(a1)-1-x]-a1[len(a1)-1])
    return np.array(vals[::-1])


class Trader():
    def __init__(self,data):
        self.data = data

    def prepare_patterns(self):
        self.hist_patterns = []
        j = self.look_back
        while j < self.index - self.look_back:
            if (self.data[j,6] - self.data[j,3]) > .0002:
                j += 10
                continue
            self.hist_patterns.append([price_scale(self.data[j-self.look_back:j+1,3]),
                                        self.data[j-self.look_back:j+1,7],
                                        np.mean(self.data[j+1:j+self.look_forward,3]) - self.data[j,3]])
            j += 3
        self.hist_patterns = np.array(self.hist_patterns)

    def long(self,length,take,stop):
        buy_price = self.data[self.index,6]
        take_price = buy_price + take
        stop_price = buy_price - stop
        end_index = np.min([data.shape[0],self.index+length])
        while self.index < end_index:
            if self.data[self.index,2] <= stop_price:
                self.money.append(self.money[-1] + (self.money[-1] * -stop * .01 / stop))
                return
            if self.data[self.index,1] > take_price:
                self.money.append(self.money[-1] + (self.money[-1] * take * .01 / stop))
                return
            self.index += 1
        self.money.append(self.money[-1] + (self.money[-1] * (self.data[self.index,3] - buy_price) * .01 / stop))

    def short(self,length,take,stop):
        sell_price = self.data[self.index,3]
        take_price = sell_price - take
        stop_price = sell_price + stop
        end_index = np.min([data.shape[0],self.index+length])
        while self.index < end_index:
            if self.data[self.index,4] >= stop_price:
                self.money.append(self.money[-1] + (self.money[-1] * -stop * .01 / stop))
                return
            if self.data[self.index,5] < take_price:
                self.money.append(self.money[-1] + (self.money[-1] * take * .01 / stop))
                return
            self.index += 1
        self.money.append(self.money[-1] + (self.money[-1] * (sell_price - self.data[self.index,6]) * .01 / stop))

    def trade(self,parameters):
        self.look_back = int(parameters[0])
        self.look_forward = int(parameters[1])
        self.sim_thresh1 = round(parameters[2],5)
        self.sim_thresh2 = round(parameters[3],5)
        self.sim_thresh3 = round(parameters[4],5)
        self.money = [10000]
        self.index = int(.9*data.shape[0])
        self.prepare_patterns()

        accuracy = []
        while self.index < data.shape[0]:
            print(self.index)
            spread = self.data[self.index,6] - self.data[self.index,3]
            if spread > .0002:
                self.index += 20
                continue
            pattern = self.data[self.index-self.look_back:self.index+1,3]
            scaled = price_scale(pattern)
            vol = self.data[self.index-self.look_back:self.index+1,7]

            matches = []
            for x in range(self.hist_patterns.shape[0]):
                p = pearsonr(scaled,self.hist_patterns[x][0])[0]
                if p > self.sim_thresh1:
                    rmse = np.sqrt(1 - mean_squared_error(scaled,self.hist_patterns[x][0]))
                    if rmse > self.sim_thresh2:
                        v = levene(vol,self.hist_patterns[x][1])[0]
                        if v < self.sim_thresh3:
                            matches.append(self.hist_patterns[x])

            matches = np.array(matches)
            if len(matches) > 1:
                fig = plt.subplots(figsize=(10,8))
                plt.subplot(211)
                plt.plot(scaled,'k')
                for match in matches:
                    plt.plot(match[0],alpha=.5)
                    if match[2] > 0:
                        plt.scatter(self.look_back+self.look_forward,match[2],c='g',alpha=.6)
                    else:
                        plt.scatter(self.look_back+self.look_forward,match[2],c='r',alpha=.6)

                    plt.scatter(self.look_back+self.look_forward+1,np.mean(matches[:,2]),c='b')
                    plt.scatter(self.look_back+self.look_forward+2,self.data[self.index+self.look_forward,3] - self.data[self.index,3],c='k')

                    plt.plot([self.look_back,self.look_back+self.look_forward],[spread,spread],'orange')
                    plt.plot([self.look_back,self.look_back+self.look_forward],[-spread,-spread],'orange')
                plt.subplot(212)
                plt.plot(vol,'k')
                for match in matches:
                    plt.plot(match[1],alpha=.5)
                plt.show()
                plt.close()



            # self.long(self.look_forward,np.max(avg_change),np.min(avg_change))
            # print('${}'.format(round(self.money[-1],2)))
            # self.short(self.look_forward,np.min(avg_change),np.max(avg_change))
            # print('${}'.format(round(self.money[-1],2)))
            self.index += 1


data = data_loader('EUR_USD','04/10/19','2100','04/24/19','2100')
m = Trader(data)

parameters = [30,10,.93,.9993,.1]
m.trade(parameters)



#
