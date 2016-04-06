import numpy as np

#Initialize data labels for david-skene
#Read known labels to this function to initialise for 
def initialize(ini_data_labels):
    label_size = len(set(ini_data_labels))
    labels = sorted(set(ini_data_labels))
    item_classes = []
    for l in ini_data_labels:
        size_list = np.zeros(label_size)
        for idx, i in enumerate(labels):
            if l == i:
                size_list[idx] = l
                item_classes.append(size_list)
    return np.array(item_classes)

#print initialize(ini_data_labels)


#read data to create responses for david-skene
def create_responses(filename): #filename in str
    row_data = []
    with open(filename) as file:
        for line in file:
            list_no = [float(val) for val in line.split(',')]
            row_data.append(list_no)

    responses = {}
    for idx,term in enumerate(zip(*row_data)): #transpose data
        user_dict = {}
        for id, val in enumerate(term):
            if val != 0:
                user_dict[id+1] = [int(val)]
        responses[idx+1] = user_dict
    return responses


#print create_responses("test_data")


