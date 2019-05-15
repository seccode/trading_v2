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
    num_clusters = 100

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

    bins = {k: [] for k in range(num_clusters)}
    outcomes = {k: [] for k in range(num_clusters)}

    X = patterns

    clusterer = AgglomerativeClustering(n_clusters=num_clusters,affinity='cosine',linkage='average')
    # clusterer = AgglomerativeClustering(n_clusters=num_clusters)
    clustering = clusterer.fit(X)

    previous_label = -1
    for j, pattern in enumerate(patterns):
        if clustering.labels_[j] == previous_label:
            continue
        bins[clustering.labels_[j]].append(pattern)
        outcomes[clustering.labels_[j]].append(endings[j])
        previous_label = clustering.labels_[j]

    for key, value in bins.items():
        value = np.array(value)
        outcomes[key] = np.array(outcomes[key])

    # plt.title('Clustering Dendrogram')
    # plot_dendrogram(clustering,labels=clustering.labels_)
    # plt.show()

    return bins, outcomes, clusterer, patterns



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
    def __init__(self,data,bins,outcomes,inductive_learner):
        self.data = data
        self.bins = bins
        self.outcomes = outcomes
        self.inductive_learner = inductive_learner

    def long(self,max,take,stop):
        end_index = np.min([self.index+max,self.data.shape[0]])
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
        end_index = np.min([self.index+max,self.data.shape[0]])
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
        self.look_back = 50
        self.look_forward = 20
        self.sim_thresh = round(parameters[0],2)
        self.match_num = int(parameters[1])
        self.mult = round(parameters[2],2)

        self.index = self.look_back
        self.money = [10000]
        while self.index < self.data.shape[0]:
            # print(self.index)
            if self.data[self.index,6] - self.data[self.index,3] > .0002:
                self.index += self.look_forward
                continue

            current_pattern = reverse_price_scale(self.data[self.index-self.look_back:self.index+1,3])
            spread = self.data[self.index,6] - self.data[self.index,3]
            index = self.inductive_learner.predict(current_pattern.reshape(1,-1))[0]

            match_pats = []
            match_outs = []
            tracker = [[],[]]
            for j, pat in enumerate(self.bins[index]):
                cs = cosine_similarity([current_pattern,pat])[0][1]
                if cs > self.sim_thresh:
                    match_pats.append(pat)
                    match_outs.append(self.outcomes[index][j])
                    tracker[0].append(dtw(current_pattern,pat))
                    tracker[1].append(match_outs[-1][-1])
            match_pats = np.array(match_pats)
            match_outs = np.array(match_outs)

            if len(match_pats) > 7:
                if np.abs(np.mean(match_outs[:,-1])) > spread*self.mult:

                    if np.mean(match_outs[:,-1]) > 0:
                        self.long(self.look_forward,np.mean(match_outs[:,-1]),np.max([-np.min(match_outs[:,-1]),.0005]))
                    else:
                        self.short(self.look_forward,-np.mean(match_outs[:,-1]),np.max([np.max(match_outs[:,-1]),.0005]))
                    # print("Account Size: ${}".format(round(self.money[-1],2)))


                    # plt.subplot(211)
                    # plt.plot(current_pattern,'k')
                    # plt.plot(range(self.look_back,self.look_back+self.look_forward),price_scale(self.data[self.index:self.index+self.look_forward,3]),'k')
                    # for x in range(len(match_pats)):
                    #     p = plt.plot(match_pats[x],alpha=.4)
                    #     plt.plot(range(self.look_back,self.look_back+self.look_forward),match_outs[x],c=p[0].get_color(),alpha=.4)
                    # plt.subplot(212)
                    # for x in range(len(tracker[0])):
                    #     plt.scatter(tracker[0][x],tracker[1][x])
                    # plt.plot(np.unique(tracker[0]), np.poly1d(np.polyfit(tracker[0], tracker[1], 1))(np.unique(tracker[0])))
                    # plt.xlim(np.min(tracker[0]),np.max(tracker[0]))
                    # plt.ylim(np.min(tracker[1]),np.max(tracker[1]))
                    # plt.show()

            self.index += 1

        return round(100 * (self.money[-1] / self.money[0] - 1),2)





train_data, train_labels = data_loader('EUR_USD','01/22/19','2100','02/12/19','2100')
test_data, test_labels = data_loader('EUR_USD','02/12/19','2100','02/18/19','2100')


print("Clustering Patterns...\n")
bins, outcomes, clusterer, X = cluster(train_data,[50,20])

print("Fitting Classifier...\n")
clf = RandomForestClassifier()
inductive_learner = InductiveClusterer(clusterer, clf).fit(X)


parameters = [.9]
print("Trading...\n")
m = Trader(test_data,bins,outcomes,inductive_learner)

bounds = np.array([[.7,.98], [2,12], [.5,5]])
best_params = bayesian_optimisation(100,m.trade,bounds)

# ROR = m.trade(parameters)















#
