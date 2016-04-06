from outlier_check import is_outlier
import numpy as np



#print zip(*db_list)

filename = "RO5p_em.csv"
row_data = []
with open(filename) as file:
    for line in file:
        list_no = [float(val) for val in line.split(',')]
        row_data.append(list_no)


col_data = zip(*row_data)


final_dict = {}
for id, term in enumerate(col_data):
    dict_a = {}
    for idx, val in enumerate(term):
        dict_a[idx+1] = val
    final_dict[id+1] = dict_a



i = 1
responses = {}
for k1, val_dict in final_dict.iteritems():
    dict__ = {}
    for k2, v in val_dict.iteritems():
        if v == 0:
            pass
        else:
            dict__[k2] = [int(v)]
    responses[i] = dict__
    i+= 1



responses = {i:responses[i] for i in responses if responses[i]!= {}}