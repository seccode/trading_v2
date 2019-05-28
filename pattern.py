
import pandas as pd
import matplotlib.pyplot as plt
from bayesianOpt import bayesian_optimisation
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from data_loader import data_loader
from scipy.stats import levene
import numpy as np


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

    def __init__(self,train_data,test_data):
        self.train_data = train_data
        self.data = test_data

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

    def long(self,max_time,take,stop):
        # Long position
        buy_price = self.data[self.index,6]
        take_price = buy_price + take
        stop_price = buy_price - stop
        end_index = np.min([self.index+max_time,self.data.shape[0]])
        while self.index < end_index:
            if self.data[self.index,2] <= stop_price:
                self.money.append(self.money[-1] - self.money[-1] * .01)
                return
            if self.data[self.index,1] > take_price:
                self.money.append(self.money[-1] + self.money[-1] * .01 * take / stop)
                return
            self.index += 1
        self.money.append(self.money[-1] + self.money[-1] * .01 * (self.data[self.index,3] - buy_price) / stop)

    def short(self,max_time,take,stop):
        # Short position
        sell_price = self.data[self.index,3]
        take_price = sell_price - take
        stop_price = sell_price + stop
        end_index = np.min([self.index+max_time,self.data.shape[0]])
        while self.index < end_index:
            if self.data[self.index,4] >= stop_price:
                self.money.append(self.money[-1] - self.money[-1] * .01)
                return
            if self.data[self.index,5] < take_price:
                self.money.append(self.money[-1] + self.money[-1] * .01 * take / stop)
                return
            self.index += 1
        self.money.append(self.money[-1] + self.money[-1] * .01 * (sell_price - self.data[self.index,6]) / stop)

    def trade(self,parameters):
        self.look_back = int(parameters[0])
        self.look_forward = int(parameters[1])
        self.thresh = round(parameters[2],5)
        self.vol_thresh = round(parameters[3],2)
        self.match_num = int(parameters[4])

        # Create patterns
        self.make_patterns()

        self.money = [10000]
        self.index = self.look_back

        current_day = -1
        # Begin trading
        while self.index < self.data.shape[0] - self.look_forward:
            # if int(self.index/1440) > current_day:
            #     current_day = int(self.index/1440)
            #     print("On day {}".format(current_day))
            # Get current pattern and outcome
            spread = self.data[self.index,6] - self.data[self.index,3]
            current_pattern = reverse_price_scale(self.data[self.index-self.look_back:self.index+1,3])
            current_outcome = price_scale(self.data[self.index:self.index+self.look_forward,3])
            current_volume = self.data[self.index-self.look_back:self.index+1,7]

            # Compare current pattern to stored patterns
            # cs = cosine_similarity(self.patterns,current_pattern.reshape(1,-1))
            cs = euclidean_distances(self.patterns,current_pattern.reshape(1,-1))

            # Find indices where pattern similarity is high
            inds = np.where(cs < np.sort(cs.flatten())[40])[0]
            if inds.shape[0] < 4:
                # Update patterns
                self.patterns = np.concatenate((self.patterns[1:,:],np.array([current_pattern])))
                self.outcomes = np.concatenate((self.outcomes[1:,:],np.array([current_outcome])))
                self.volumes = np.concatenate((self.volumes[1:,:],np.array([current_volume])))
                self.index += 1
                continue

            # Remove duplicate patterns and patterns with non-matching volumes
            new_inds = []
            j = 0
            while j < inds.shape[0] - self.look_back:
                if ((j == 0) or (j != 0 and inds[j] - inds[j-1] > self.look_back/2)) and \
                    levene(current_volume,self.volumes[inds[j]])[1] > self.vol_thresh:

                    new_inds.append(inds[j])
                j += 1
            inds = np.array(new_inds)

            # Plot patterns and open position if outcomes are good
            if inds.shape[0] > self.match_num and np.abs(np.median(self.outcomes[inds,-1])) > self.thresh:
                # plt.plot(current_pattern,'k')
                # plt.plot(range(self.look_back,self.look_back+self.look_forward),current_outcome,'k')
                # for ind in inds:
                #     p = plt.plot(self.patterns[ind],alpha=.4)
                #     plt.plot(range(self.look_back,self.look_back+self.look_forward),self.outcomes[ind],alpha=.4,c=p[0].get_color())
                # plt.show()

                if np.median(self.outcomes[inds,-1]) > 0:
                    self.long(self.look_forward,np.median(self.outcomes[inds,-1]),-np.min(self.outcomes[inds])+2*spread)
                else:
                    self.short(self.look_forward,-np.median(self.outcomes[inds,-1]),np.max(self.outcomes[inds])+2*spread)
                # print("Account Size: ${}".format(round(self.money[-1],2)))

            # Update patterns
            self.patterns = np.concatenate((self.patterns[1:,:],np.array([current_pattern])))
            self.outcomes = np.concatenate((self.outcomes[1:,:],np.array([current_outcome])))
            self.volumes = np.concatenate((self.volumes[1:,:],np.array([current_volume])))
            self.index += 1

        return round(100 * (self.money[-1] / self.money[0] - 1),2)


train_data, _ = data_loader('EUR_USD','01/02/19','2100','02/25/19','2100')
test_data, _ = data_loader('EUR_USD','02/25/19','2100','03/11/19','2100')
m = Trader(train_data,test_data)

bounds = np.array([ [30,120], [10,60], [.0001,.0015], [.01,.25], [1,15] ])
best_params = bayesian_optimisation(200,m.trade,bounds)

# parameters = [71, 15, 0.00036, 0.07, 8]
# ror = m.trade(parameters)
# print("Rate of Return: {}%".format(ror))









#
