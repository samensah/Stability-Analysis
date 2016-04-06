from kmeans_responses import responses, num_users, num_questions
import numpy as np
import csv

#print responses

db_list = []
for key, val in responses.iteritems():
    column = [0]*num_users
    for i in range(1, len(column)+1):
        if i in val:
            for k, v in val.iteritems():
                if k == i:
                    column[i-1] = v[0]
    db_list.append(column)


#print zip(*db_list)


myfile = open('./labelled_data/responses_out100_3.5.csv', 'wb')
for term in zip(*db_list):
    wr = csv.writer(myfile, delimiter=',', quoting=csv.QUOTE_ALL)
    wr.writerow(term)

