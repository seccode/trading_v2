'''
Pattern Strategy
'''
from scipy.stats import levene
from scipy.stats import pearsonr
from bayesianOpt import bayesian_optimisation
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import euclidean, braycurtis
from tslearn.metrics import dtw as dynamic_time_warp
from tslearn.metrics import gak as global_alignment_kernel

from data_loader import data_loader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

metrics = ['Pearson','Cosine','Volume','Bray Curtis','Euclidean', 'Dynamic Time Warp', 'Global Alignment Kernel']

def price_scale(a1):
    vals = [0]
    for x in range(1,len(a1)):
        vals.append(np.mean(a1[len(a1)-1-x]) - np.mean(a1[len(a1)-1]))
    return np.array(vals[::-1])

def kelly(take,stop,prob):
    '''Kelly Criterion Formula'''
    return (prob * (take/stop + 1) - 1) / (take/stop)



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
                                        np.median(self.data[j+1:j+self.look_forward,3]) - self.data[j,3]])
            j += 3
        self.hist_patterns = np.array(self.hist_patterns)

    def long(self,length,take,stop):
        buy_price = self.data[self.index,6]
        take_price = buy_price + take
        stop_price = buy_price - stop
        end_index = np.min([data.shape[0],self.index+length])
        while self.index < end_index:
            # if self.data[self.index,2] <= stop_price:
            #     self.money.append(self.money[-1] + (self.money[-1] * -stop * .01 / stop))
            #     return
            if self.data[self.index,1] > take_price:
                self.money.append(self.money[-1] + (self.money[-1] * take * .01 / stop))
                return
            # if self.data[self.index,1] - stop_price > stop:
            #     stop_price = self.data[self.index,1] - stop
            self.index += 1
        self.money.append(self.money[-1] + (self.money[-1] * (self.data[self.index,3] - buy_price) * .01 / stop))

    def short(self,length,take,stop):
        sell_price = self.data[self.index,3]
        take_price = sell_price - take
        stop_price = sell_price + stop
        end_index = np.min([data.shape[0],self.index+length])
        while self.index < end_index:
            # if self.data[self.index,4] >= stop_price:
            #     self.money.append(self.money[-1] + (self.money[-1] * -stop * .01 / stop))
            #     return
            if self.data[self.index,5] < take_price:
                self.money.append(self.money[-1] + (self.money[-1] * take * .01 / stop))
                return
            # if stop_price - self.data[self.index,5] > stop:
            #     stop_price = self.data[self.index,5] + stop
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
        start_i = int(.9*data.shape[0])
        self.prepare_patterns()
        curr_day = 1

        tracker = []
        print('On day 1')
        while self.index < data.shape[0]:

            if round((self.index - start_i) / 1440) > curr_day:
                print('On day: {}'.format(round((self.index - start_i) / 1440)))
                curr_day += 1

            spread = self.data[self.index,6] - self.data[self.index,3]
            if spread > .0002:
                self.index += 20
                continue

            scaled = price_scale(self.data[self.index-self.look_back:self.index+1,1:4])
            vol = self.data[self.index-self.look_back:self.index+1,7]

            matches = []
            mean_outcome = np.median(self.data[self.index+1:self.index+self.look_forward,3]) - self.data[self.index,3]
            for x in range(self.hist_patterns.shape[0]):

                cs = cosine_similarity([scaled,self.hist_patterns[x][0]])[0][1]
                if cs > self.sim_thresh1:
                    p = pearsonr(scaled,self.hist_patterns[x][0])[0]
                    if p > self.sim_thresh2:
                        v = levene(vol,self.hist_patterns[x][1])[0]
                        if v < self.sim_thresh3:
                            dtw = dynamic_time_warp(scaled,self.hist_patterns[x][0])
                            bc = braycurtis(scaled,self.hist_patterns[x][0])
                            ec = euclidean(scaled,self.hist_patterns[x][0])
                            gak = global_alignment_kernel(scaled,self.hist_patterns[x][0])
                            if self.hist_patterns[x][2] > 0:
                                if mean_outcome > 0:
                                    tracker.append([p,cs,v,bc,ec,dtw,gak,1])
                                else:
                                    tracker.append([p,cs,v,bc,ec,dtw,gak,0])
                            else:
                                if mean_outcome < 0:
                                    tracker.append([p,cs,v,bc,ec,dtw,gak,1])
                                else:
                                    tracker.append([p,cs,v,bc,ec,dtw,gak,0])

                            matches.append(self.hist_patterns[x])


            matches = np.array(matches)
            if len(matches) > 4:

                fig, ax = plt.subplots(figsize=(8,6))
                plt.subplot(211)
                plt.plot(scaled,'k')
                after = self.data[self.index:self.index+self.look_forward,3]
                new = [0]
                for x in range(1,len(after)):
                    new.append(after[x] - after[0])

                # plt.plot(range(self.look_back,self.look_back+self.look_forward),new,'k')

                outcomes = []
                for match in matches:
                    if match[2] > 0:
                        plt.scatter(self.look_back+self.look_forward,match[2],c='green',alpha=.4)
                        outcomes.append(match[2])
                    else:
                        plt.scatter(self.look_back+self.look_forward,match[2],c='red',alpha=.4)
                        outcomes.append(match[2])
                    plt.plot(match[0],alpha=.5)
                    plt.plot([self.look_back,self.look_back+self.look_forward],[spread,spread],'orange')
                    plt.plot([self.look_back,self.look_back+self.look_forward],[-spread,-spread],'orange')

                plt.scatter(self.look_back+self.look_forward+1,np.median(outcomes),c='blue')
                plt.scatter(self.look_back+self.look_forward+2,np.max(self.data[self.index+1:self.index+self.look_forward,1]) - self.data[self.index,3],c='black')
                plt.scatter(self.look_back+self.look_forward+2,np.min(self.data[self.index+1:self.index+self.look_forward,2]) - self.data[self.index,3],c='black')

                plt.subplot(212)
                plt.plot(vol,'k')
                for match in matches:
                    plt.plot(match[1],alpha=.5)
                # plt.show()

                # plt.show()
                if np.median(outcomes) > 2*spread:

                    self.long(self.look_forward,(np.median(outcomes)+spread)/4,np.max([.0004,np.max(np.abs(matches[:,2]))]))
                    print('Long\n${}\n'.format(round(self.money[-1],2)))
                    fig.suptitle('Long_'+str(self.index)+' '+str(self.money[-1] - self.money[-2]))
                    plt.savefig('plots/Long_'+str(self.index)+'.png')
                    plt.show()

                elif np.median(outcomes) < -2*spread:

                    self.short(self.look_forward,-(np.median(outcomes)+spread)/4,np.max([.0004,np.max(matches[:,2])]))
                    print('Short\n${}\n'.format(round(self.money[-1],2)))
                    fig.suptitle('Short_'+str(self.index)+' '+str(self.money[-1] - self.money[-2]))
                    plt.savefig('plots/Short_'+str(self.index)+'.png')
                    plt.show()
                else:
                    plt.close()
                    self.index += 1
                    continue
                plt.close()

                new_tracker = np.array(tracker)
                for i, metric in enumerate(metrics):
                    plt.title(metric)
                    for x in range(len(tracker)):
                        plt.scatter(tracker[x][i],tracker[x][-1])

                    plt.plot(np.unique(new_tracker[:,i]), np.poly1d(np.polyfit(new_tracker[:,i], new_tracker[:,-1], 1))(np.unique(new_tracker[:,i])))
                    plt.xlim(np.min(new_tracker[:,i]),np.max(new_tracker[:,i]))
                    plt.ylim(np.min(new_tracker[:,-1]),np.max(new_tracker[:,-1]))
                    plt.savefig(metric+'.png')
                    plt.close()


            self.index += 1


data = data_loader('EUR_USD','02/06/19','2100','03/20/19','2100')
m = Trader(data)

parameters = [50,20,.95,.8,1]
m.trade(parameters)



#
