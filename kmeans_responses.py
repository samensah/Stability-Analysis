author = 'samuel'
"""
K-Means Clustering
"""

import random
#from outlier_check import is_outlier
from itertools import groupby
import numpy as np
from sklearn.cluster import KMeans



#list_no = [float(val) for val in line.split(',')], define delimeter
#define input file

def load_datapoints(filename):
    data_ = []
    with open(filename) as file:
        for line in file:
            #define delimeter in dataRT input file
            list_no = [float(val) for val in line.split(',')]
            #list_no = [float(val) for val in line.split()]
            data_.append(list_no)
    data_set = list(zip(*data_))
    data_set_ = [list(dat) for dat in data_set]
    for term in data_set_:
        for idx, point in enumerate(term):
            if point == -1:
                term[idx] = 0.
    num_users, num_questions = len(data_set[0]), len(data_set)
    data_set = [tuple(dat) for dat in data_set_]
    return data_set, num_users, num_questions

#define dataRT input
data_set, num_users, num_questions = load_datapoints("dataRT_csv_new.csv")

def is_outlier(points, thresh=3.5):

    if len(points.shape) == 1:
        points = points[:,None] #make a list of elements in points
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation
    return modified_z_score > thresh





def return_clusters(data, centroids):
    mylist = []
    min_ss = []
    data_dict = {}
    for idx, item in enumerate(data):
        data_dict.update({idx:item})
        distance_list = []
        if item != 0:
            for cent in centroids:
                sum_squares = abs(item - cent)
                distance_list.append([cent, [idx, item], sum_squares])
            data_fig = [num[-1] for num in distance_list]
            min_ss.append(min(data_fig))
            mylist.append(distance_list[data_fig.index(min(data_fig))][0:2])
    clusters = [list(v) for l,v in groupby(sorted(mylist, key=lambda x:x[0]), lambda x: x[0])]

    return data_dict, sum(min_ss), clusters




def kmeans(data, k):
    data_rem = filter(lambda a: a != 0 and a != -1, data)
    data_rem = zip(data_rem, np.zeros(len(data_rem)))
    if len(data_rem) < k:
        k_means = KMeans(init='k-means++', n_clusters=len(data_rem), n_init=10)
        k_means.fit(data_rem)
        k_means_cluster_centers = k_means.cluster_centers_
        centroids = [k[0] for k in k_means_cluster_centers]
    else:
        k_means = KMeans(init='k-means++', n_clusters=k, n_init=10)
        k_means.fit(data_rem)
        k_means_cluster_centers = k_means.cluster_centers_
        centroids = [k[0] for k in k_means_cluster_centers]
    data_dict, ss, clusters = return_clusters(data, centroids)
    return data_dict, clusters, sorted(centroids) #centroid and data





def def_dict(centroids, clusters):
    cluster_dict = {}
    for idx, cent in enumerate(centroids):
        for term in clusters:
            for point in term:
                if cent == point[0]:
                    cluster_dict.update({point[1][0]+1: [idx+1]}) #place value in []
    return cluster_dict





def dict_val_grp(centroids, cluster_dict):
    group_val = {}
    for i in range(len(centroids)):
        value_center = []
        for k, v in cluster_dict.iteritems():
            if v == [i+1]: #place value in []
                value_center.append(k)
        group_val.update({i+1:value_center})
    return group_val



def remove_outlier(data_dict, cluster_dict, dict_val, thresh=1.5):
    #cluster_dict
    for k, v in dict_val.iteritems():
        x = [data_dict[i-1] for i in v]
        truth = is_outlier(np.array(x), thresh)
        for j in zip(v, truth):
            if j[1] == True:

                #remove outlier from dictionary
                cluster_dict.pop(j[0])

                #assign outlier to label 4
                #cluster_dict[j[0]] = [4]
    return cluster_dict



def gen_category(data_set, k=3, thresh=3.5):
    data_categorized = {}
    for id, dat in enumerate(data_set):
        data_dict, clusters, centroids = kmeans(dat, k)
        cluster_dict = def_dict(centroids, clusters)
        dict_val = dict_val_grp(centroids, cluster_dict)
        #print(cluster_dict) #category with outliers
        dat_cat = remove_outlier(data_dict, cluster_dict, dict_val, thresh)
        data_categorized.update({id+1:dat_cat})
    return data_categorized


responses = gen_category(data_set, k=3, thresh=3.5)

#print(is_outlier(np.array([1,2,3,4,5,6,15]), thresh=1.5))