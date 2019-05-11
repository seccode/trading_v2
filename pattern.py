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
        vals.append(np.mean(a1[len(a1)-1-x]) - np.mean(a1[len(a1)-1]))
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
            self.hist_patterns.append([price_scale(self.data[j-self.look_back:j+1,1:4]),
                                        self.data[j-self.look_back:j+1,7],
                                        np.max(self.data[j+1:j+self.look_forward,1]) - self.data[j,3],
                                        self.data[j,3] - np.min(self.data[j+1:j+self.look_forward,2])])
            j += 3
        self.hist_patterns = np.array(self.hist_patterns)

    def long(self,length,take,stop):
        print(take)
        print(stop)
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
        print(take)
        print(stop)
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
            # print(self.index)
            spread = self.data[self.index,6] - self.data[self.index,3]
            if spread > .0002:
                self.index += 20
                continue

            scaled = price_scale(self.data[self.index-self.look_back:self.index+1,1:4])
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
            if len(matches) > 2:


                fig, ax = plt.subplots(figsize=(8,6))
                plt.subplot(211)
                plt.plot(scaled,'k')
                after = self.data[self.index:self.index+self.look_forward,3]
                new = [0]
                for x in range(1,len(after)):
                    new.append(after[x] - after[0])

                plt.plot(range(self.look_back,self.look_back+self.look_forward),new)

                outcomes = []
                for match in matches:
                    if match[2] > match[3]:
                        plt.scatter(self.look_back+self.look_forward,match[2],c='green',alpha=.4)
                        outcomes.append(match[2])
                    else:
                        plt.scatter(self.look_back+self.look_forward,-match[3],c='red',alpha=.4)
                        outcomes.append(-match[3])
                    plt.plot(match[0],alpha=.5)
                    plt.plot([self.look_back,self.look_back+self.look_forward],[spread,spread],'orange')
                    plt.plot([self.look_back,self.look_back+self.look_forward],[-spread,-spread],'orange')
                plt.scatter(self.look_back+self.look_forward+1,np.mean(outcomes),c='blue')
                plt.scatter(self.look_back+self.look_forward+2,np.max(self.data[self.index+1:self.index+self.look_forward,1]) - self.data[self.index,3],c='black')
                plt.scatter(self.look_back+self.look_forward+2,np.min(self.data[self.index+1:self.index+self.look_forward,2]) - self.data[self.index,3],c='black')

                plt.subplot(212)
                plt.plot(vol,'k')
                for match in matches:
                    plt.plot(match[1],alpha=.5)
                # plt.show()

                if np.mean(outcomes) > 2*spread:
                    self.long(self.look_forward,np.mean(outcomes),np.max([.0004,np.max(np.abs(matches[:,3]))]))
                    print('Long\n${}\n'.format(round(self.money[-1],2)))
                    fig.suptitle('Long_'+str(self.index)+' '+str(self.money[-1] - self.money[-2]))
                    plt.savefig('plots/Long_'+str(self.index)+'.png')
                elif np.mean(outcomes) < -2*spread:
                    self.short(self.look_forward,-np.mean(outcomes),np.max([.0004,np.max(matches[:,2])]))
                    print('Short\n${}\n'.format(round(self.money[-1],2)))
                    fig.suptitle('Short_'+str(self.index)+' '+str(self.money[-1] - self.money[-2]))
                    plt.savefig('plots/Short_'+str(self.index)+'.png')
                plt.close()



            self.index += 1


data = data_loader('EUR_USD','01/08/19','2100','04/19/19','2100')
m = Trader(data)

parameters = [50,20,.95,.9997,.08]
m.trade(parameters)



#
