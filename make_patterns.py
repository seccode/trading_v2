import csv
import numpy as np
from data_loader import data_loader
import matplotlib.pyplot as plt

def reverse_price_scale(a1):
    vals = [0]
    for x in range(1,len(a1)):
        vals.append(100000 * (np.mean(a1[len(a1)-1-x]) - np.mean(a1[len(a1)-1])))
    return vals[::-1]

def price_scale(a1):
    vals = [0]
    for x in range(1,len(a1)):
        vals.append(100000 * (np.mean(a1[x]) - np.mean(a1[0])))
    return vals


data, labels = data_loader(all=True)
# data, labels = data_loader('EUR_USD','04/10/19','2100','04/11/19','2100')

look_back = 50
look_forward = 20

patterns = []
i = look_back
while i < data.shape[0] - look_forward:
    patterns.append([labels[i],data[i,0],reverse_price_scale(data[i-look_back:i+1,3]),price_scale(data[i:i+look_forward,3])])
    i += 1


with open('all_patterns.csv','w') as csvFile:
    csv_writer = csv.writer(csvFile)
    for pattern in patterns:
        csv_writer.writerow(pattern)

csvFile.close()

























#
