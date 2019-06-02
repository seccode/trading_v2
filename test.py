
import matplotlib.pyplot as plt
from data_loader import data_loader
import numpy as np

class Trader():
    def __init__(self,data):
        self.data = data

    def order_position(self,type='long',price=1.0000,take=.0015,stop=.0015,size=10000,market_order=True):
        self.positions[self.index] = {'type':type,
                                    'enter_price':price,
                                    'take':price + take,
                                    'stop':price - stop,
                                    'size':size,
                                    'open':market_order}

    def check_positions(self):
        remove_keys = []
        for key, trade in self.positions.items():
            if trade['type'] == 'long':
                if trade['open']:
                    if self.data[self.index,2] < trade['stop']:
                        self.update_money(trade['stop'] - trade['enter_price'],trade['size'])
                        remove_keys.append(key)
                    elif self.data[self.index,1] > trade['take']:
                        self.update_money(trade['take'] - trade['enter_price'],trade['size'])
                        remove_keys.append(key)
            else:
                if trade['open']:
                    if self.data[self.index,4] > trade['stop']:
                        self.update_money(trade['enter_price'] - trade['stop'],trade['size'])
                        remove_keys.append(key)
                    elif self.data[self.index,5] < trade['take']:
                        self.update_money(trade['enter_price'] - trade['take'],trade['size'])
                        remove_keys.append(key)
        for key in remove_keys:
            del self.positions[key]

    def update_money(self,profit,size):
        self.money.append(self.money[-1] + (profit * size))

    def trade(self):
        self.index = 1440
        self.money = [10000]
        self.positions = {}

        while self.index < self.data.shape[0]:
            spread = self.data[self.index,6] - self.data[self.index,3]
            self.check_positions()

            if self.data[self.index,0] == 100:
                self.order_position(type='long',
                                    price=self.data[self.index,6],
                                    take=5*spread,
                                    stop=10*spread,
                                    size=10000,
                                    market_order=True)

            self.index += 1
        return round(100 * (self.money[-1] / self.money[0] - 1),2)


data, _ = data_loader('USD_JPY','01/03/19','2100','05/20/19','2100')
m = Trader(data)
ror = m.trade()
print("Rate of Return: {}%".format(ror))




#
