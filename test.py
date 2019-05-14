
import numpy as np
import matplotlib.pyplot as plt
from data_loader import data_loader

data = data_loader('EUR_USD','05/09/19','2100','05/15/19','2100')

plt.hist(data[:,3],500)
plt.show()











#
