'''
Kalman Filter Strategy
'''

import numpy as np
import matplotlib.pyplot as plt
from data_loader import data_loader
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from bayesianOpt import bayesian_optimisation
from pykalman import KalmanFilter
import pandas as pd
import csv

class trader():
    def __init__(self,data):
        self.data = data

    def long(self,length):
        buy_price = self.data[self.index,6]
        end_index = np.min([self.data.shape[0],self.index+length])
        while self.index < end_index:
            self.index += 1
        self.money.append(self.money[-1] + (self.money[-1] * (self.data[self.index,3] - buy_price) * .01 /.001))
        print(self.money[-1])
        return

    def short(self,length):
        sell_price = self.data[self.index,3]
        end_index = np.min([self.data.shape[0],self.index+length])
        while self.index < end_index:
            self.index += 1
        self.money.append(self.money[-1] + (self.money[-1] * (sell_price - self.data[self.index,6]) * .01 /.001))
        print(self.money[-1])
        return

    def trade(self,parameters):
        self.look_back = int(parameters[0])
        self.money = [10000]
        self.index = self.look_back

        # plt.ion()
        # plt.show()
        pred_dist = [0] * self.look_back

        while self.index < self.data.shape[0]:
            try:

                x = np.linspace(0,self.look_back,self.look_back+1)
                data_set = data[self.index - self.look_back:self.index+1,3]

                kf = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),
                                transition_covariance=0.01 * np.eye(2))


                # states_pred = kf.em(data_set).smooth(data_set)[0]
                x = x[50:]
                states_pred = kf.smooth(data_set)[0][50:]
                data_set = data_set[50:]



                separation = data_set - states_pred[:,0]
                model = ARIMA(separation,order=(5,1,0))
                # model = ARIMA(states_pred[:,0],order=(5,1,0))
                model_fit = model.fit(disp=0)
                output = model_fit.forecast(5)

                if separation[-1] - output[0][0] > .00015:
                    self.short(2)
                elif separation[-1] - output[0][0] < -.00015:
                    self.long(2)



                # plt.close()
                # fig, ax = plt.subplots(figsize=(10,6))
                # plt.subplot(311)
                # plt.plot(data_set, color='b',label='observations')
                # plt.plot(states_pred[:, 0],color='r',label='position est.')
                # plt.legend(loc='lower left')
                #
                # plt.subplot(312)
                # # plt.plot(states_pred[:, 1],color='g',label='velocity est.')
                # plt.plot(separation,color='g',label='separation')
                # plt.plot([0,len(separation)+len(output)],[0,0])
                # # plt.plot([len(separation),len(separation)+len(output)],[np.mean(output[0]),np.mean(output[0])],'r')
                # for x in range(len(output[0])):
                #     plt.plot(len(separation)+1+x,output[0][x],'r.')
                # plt.legend(loc='lower left')
                #
                # pred_dist.append(separation[-1] - output[0][0])
                # plt.subplot(313)
                # plt.plot(pred_dist[-self.look_back+50:])
                # plt.plot([0,len(separation)+len(output)],[.00015,.00015])
                # plt.plot([0,len(separation)+len(output)],[-.00015,-.00015])
                #
                # # plt.show()
                # plt.pause(.001)
            except:
                pass
            self.index += 1


data = data_loader('EUR_USD','03/05/19','2100','05/02/19','2100')
m = trader(data)
m.trade([200])









#
