'''
Machine Learning for pattern strategy
'''
import time
from tslearn.metrics import dtw
from data_loader import data_loader
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from bayesianOpt import bayesian_optimisation
import matplotlib.pyplot as plt
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

def plot_dendrogram(model, **kwargs):
    children = model.children_
    distance = np.arange(children.shape[0])
    no_of_observations = np.arange(2, children.shape[0]+2)
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
    dendrogram(linkage_matrix, **kwargs)

def cluster(train_data,parameters):
    patterns = []
    endings = []
    look_back = int(parameters[0])
    look_forward = int(parameters[1])
    num_clusters = 500

    i = look_back
    while i < train_data.shape[0] - look_forward:
        if train_data[i,6] - train_data[i,3] > .0002:
            i += look_forward
            continue
        patterns.append(reverse_price_scale(train_data[i-look_back:i+1,3]))
        endings.append(price_scale(train_data[i:i+look_forward,3]))
        i += 1
    patterns = np.array(patterns)
    endings = np.array(endings)

    X = patterns

    # clusterer = AgglomerativeClustering(n_clusters=num_clusters,affinity='cosine',linkage='average')
    clusterer = AgglomerativeClustering(n_clusters=num_clusters)
    clustering = clusterer.fit(X)

    bins = {k: [] for k in range(len(clustering.labels_))}
    outcomes = {k: [] for k in range(len(clustering.labels_))}
    previous_label = -1
    for j, pattern in enumerate(patterns):
        if clustering.labels_[j] == previous_label:
            continue
        bins[clustering.labels_[j]].append(pattern)
        outcomes[clustering.labels_[j]].append(endings[j])
        previous_label = clustering.labels_[j]

    good_bins = []
    for key, value in bins.items():
        value = np.array(value)
        outcomes[key] = np.array(outcomes[key])
        if len(value) > 3 and (np.max(outcomes[key][:,-1]) < 0 or np.min(outcomes[key][:,-1]) > 0):
        # if len(value) > 3 and np.abs(np.median(outcomes[key][:,-1])) > .0002:
            good_bins.append(key)

    # plt.title('Clustering Dendrogram')
    # plot_dendrogram(clustering,labels=clustering.labels_)
    # plt.show()
    return bins, outcomes, clusterer, patterns, good_bins


class InductiveClusterer(BaseEstimator):
    def __init__(self, clusterer, classifier):
        self.clusterer = clusterer
        self.classifier = classifier

    def fit(self, X, y=None):
        self.clusterer_ = clone(self.clusterer)
        self.classifier_ = clone(self.classifier)
        y = self.clusterer_.fit_predict(X)
        self.classifier_.fit(X, y)
        return self

    @if_delegate_has_method(delegate='classifier_')
    def predict(self, X):
        return self.classifier_.predict(X)

    @if_delegate_has_method(delegate='classifier_')
    def decision_function(self, X):
        return self.classifier_.decision_function(X)


class Trader():
    def __init__(self,data,bins,outcomes,inductive_learner,good_bins):
        self.data = data
        self.bins = bins
        self.outcomes = outcomes
        self.inductive_learner = inductive_learner
        self.good_bins = good_bins

    def long(self,max,take,stop):
        end_index = np.min([self.index+max,self.data.shape[0]-1])
        buy_price = self.data[self.index,6]
        take_price = buy_price + take
        stop_price = buy_price - stop
        while self.index < end_index:
            if self.data[self.index,2] <= stop_price:
                self.money.append(self.money[-1] + (self.money[-1] * -.01))
                return
            if self.data[self.index,1] > take_price:
                self.money.append(self.money[-1] + (self.money[-1] * .01 * take) / stop)
                return
            self.index += 1
        self.money.append(self.money[-1] + (self.money[-1] * .01 * (self.data[self.index,3] - buy_price)) / stop)

    def short(self,max,take,stop):
        end_index = np.min([self.index+max,self.data.shape[0]-1])
        sell_price = self.data[self.index,3]
        take_price = sell_price - take
        stop_price = sell_price + stop
        while self.index < end_index:
            if self.data[self.index,4] >= stop_price:
                self.money.append(self.money[-1] + (self.money[-1] * -.01))
                return
            if self.data[self.index,5] < take_price:
                self.money.append(self.money[-1] + (self.money[-1] * .01 * take) / stop)
                return
            self.index += 1
        self.money.append(self.money[-1] + (self.money[-1] * .01 * (sell_price - self.data[self.index,6])) / stop)

    def trade(self,parameters):
        self.look_back = int(parameters[0])
        self.look_forward = int(parameters[1])

        self.index = self.look_back
        self.money = [10000]
        while self.index < self.data.shape[0] - self.look_forward:
            # print(self.index)
            if self.data[self.index,6] - self.data[self.index,3] > .0002:
                self.index += self.look_forward
                continue

            current_pattern = reverse_price_scale(self.data[self.index-self.look_back:self.index+1,3])
            spread = self.data[self.index,6] - self.data[self.index,3]
            bin_num = self.inductive_learner.predict(current_pattern.reshape(1,-1))[0]
            if not bin_num in self.good_bins:
                self.index += 1
                continue

            # plt.plot(current_pattern,'k')
            # plt.plot(range(self.look_back,self.look_back+self.look_forward),price_scale(self.data[self.index:self.index+self.look_forward,3]),'k')
            # for x in range(len(self.bins[bin_num])):
            #     p = plt.plot(self.bins[bin_num][x],alpha=.4)
            #     plt.plot(range(self.look_back,self.look_back+self.look_forward),self.outcomes[bin_num][x],c=p[0].get_color(),alpha=.4)
            # plt.show()

            if np.mean(self.outcomes[bin_num][:,-1]) > spread:
                self.long(self.look_forward,np.mean(self.outcomes[bin_num][:,-1]),-np.min(self.outcomes[bin_num])+3*spread)
            elif np.mean(self.outcomes[bin_num][:,-1]) < -spread:
                self.short(self.look_forward,-np.mean(self.outcomes[bin_num][:,-1]),np.max(self.outcomes[bin_num])+3*spread)
            # print("Account Size: ${}".format(round(self.money[-1],2)))

            self.index += 1

        return round(100 * (self.money[-1] / self.money[0] - 1),2)




# train_data, train_labels = data_loader('EUR_USD','07/02/18','2100','07/23/18','2100')
# test_data, test_labels = data_loader('EUR_USD','07/23/18','2100','07/30/18','2100')
train_data, train_labels = data_loader('EUR_USD','01/28/19','2100','02/25/19','2100')
test_data, test_labels = data_loader('EUR_USD','02/25/19','2100','03/04/19','2100')
def optimize(parameters):
    '''
    Parameters:
        * Bin numbers for clustering
        * Look back length
        * Look forward length
    '''
    bin_num = int(parameters[0])
    look_back = int(parameters[1])
    look_forward = int(parameters[2])

    tot = []
    for x in range(3):
        # print("Clustering Patterns...\n")
        bins, outcomes, clusterer, X, good_bins = cluster(train_data,[look_back,look_forward])

        # print("Fitting Classifier...\n")
        clf = RandomForestClassifier()
        inductive_learner = InductiveClusterer(clusterer, clf).fit(X)


        params = [look_back,look_forward]
        # print("Trading...\n")
        m = Trader(test_data,bins,outcomes,inductive_learner,good_bins)

        ROR = m.trade(params)
        tot.append(ROR)
    return round(np.mean(tot),2)



bounds = np.array([[100,2000], [20,120], [10,60]])
best_params = bayesian_optimisation(100,optimize,bounds)
print(best_params)


# ROR = optimize([1000,30,50])
# print(ROR)










#
