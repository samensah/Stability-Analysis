import numpy as np


filename = "reliability.txt"
a = {}
with open(filename) as file:
    for line in file:
        list0, list1 = [float(val) for val in line.split()]
        a.update({int(list0): list1})
#print(my_list)


vals = sorted([value for (key, value) in sorted(a.items())])
idx = np.array(sorted(a, key=a.__getitem__, reverse=True)) #good index sorted
#print idx


id = [i+1 for i in range(len(a))]
#print id



here = []
k=1
for i in idx:
    for j in range(1, len(a)+1):
        if i == j:
            here.append([k,i,a[i]])
    k+=1
print here




