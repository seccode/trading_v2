'''
Kalman Filter Strategy
'''

import warnings
import matplotlib.pyplot as plt
from data_loader import data_loader
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from bayesianOpt import bayesian_optimisation
from pykalman import KalmanFilter
import numpy as np

def kelly(take,stop,prob):
    '''Kelly Criterion Formula'''
    return (prob*take/stop - (1 - prob)) / (take/stop)

class Trader():
    def __init__(self,data):
        self.data = data

    def long(self,length,take,stop):
        buy_price = self.data[self.index,6]
        stop_price = buy_price - stop
        take_price = buy_price + take
        end_index = np.min([self.data.shape[0],self.index+length])
        while self.index < end_index:
            if self.data[self.index,2] <= stop_price:
                self.money.append(self.money[-1] + (self.money[-1] * -stop * .01 /stop))
                return
            if self.data[self.index,1] > take_price:
                self.money.append(self.money[-1] + (self.money[-1] * take * .01 /stop))
                return
            self.index += 1
        self.money.append(self.money[-1] + (self.money[-1] * (self.data[self.index,3] - buy_price) * .01 /stop))
        return

    def short(self,length,take,stop):
        sell_price = self.data[self.index,3]
        stop_price = sell_price + stop
        take_price = sell_price - take
        end_index = np.min([self.data.shape[0],self.index+length])
        while self.index < end_index:
            if self.data[self.index,4] >= stop_price:
                self.money.append(self.money[-1] + (self.money[-1] * -stop * .01 /stop))
                return
            if self.data[self.index,5] < take_price:
                self.money.append(self.money[-1] + (self.money[-1] * take * .01 /stop))
                return
            self.index += 1
        self.money.append(self.money[-1] + (self.money[-1] * (sell_price - self.data[self.index,6]) * .01 /stop))
        return

    def trade(self,parameters):
        self.max_time = int(parameters[0])
        self.take_mult = round(parameters[1],2)
        self.stop_mult = round(parameters[2],2)
        self.look_back = 200
        self.money = [10000]
        self.index = self.look_back

        while self.index < self.data.shape[0]:
            current_ror = round(100 * (self.money[-1]/self.money[0] - 1),2)
            if current_ror < -20:
                return current_ror

            data_set = data[self.index - self.look_back:self.index+1,3]

            kf = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),
                            transition_covariance=0.01 * np.eye(2))

            # states_pred = kf.em(data_set).smooth(data_set)[0]
            states_pred = kf.smooth(data_set)[0][50:]
            data_set = data_set[50:]

            separation = data_set - states_pred[:,0]
            model = ARIMA(separation,order=(5,1,0))
            model_fit = model.fit(disp=0,method='mle')
            output = model_fit.forecast(5)

            slope = states_pred[-1,0] - states_pred[-2,0]
            projected_position = states_pred[-1,0] + 3*slope
            dist = separation[-1] - output[0][0]

            spread = self.data[self.index,6] - self.data[self.index,3]
            if dist > .25*spread:
                if data_set[-1]-dist > projected_position:
                    # print("Short")
                    self.short(self.max_time,self.take_mult*abs(dist),self.stop_mult*abs(dist))

            elif dist < -.25*spread:
                if data_set[-1]-dist < projected_position:
                    # print("Long")
                    self.long(self.max_time,self.take_mult*abs(dist),self.stop_mult*abs(dist))

            self.index += 1

        return round(100 * (self.money[-1]/self.money[0] - 1),2)



data = data_loader('EUR_USD','03/20/19','2100','03/21/19','2100')
m = Trader(data)

def backtest():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        ror = m.trade([20,.5,5])
        print(ror)


def optimize():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        bounds = np.array([[5,30], [.2,2], [2,6]])
        best_params = bayesian_optimisation(200,m.trade,bounds)
        print(best_params)

backtest()
optimize()






#
