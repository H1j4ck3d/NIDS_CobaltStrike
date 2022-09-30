import os
import pandas as pd

columns = ['src_ip','src_p','dst_ip', 'dst_p', 'proto', 'dur', 'nses', 'cltmoredata', 'clt1stappdata', 'dur_mean','dur_std', 'int_mean', 'int_std']
train_folder = 'Dataset/group-dataset/logs/train'
test_folder = 'Dataset/group-dataset/logs/test'

mal = ['172.93.179.196', '52.18.235.51',  '23.82.141.136', '179.60.150.125', '108.177.235.172', '172.93.201.12', '23.106.215.100',
       '172.241.27.237', '139.60.161.45', '139.60.160.8',  '5.149.249.187', '103.207.42.11', '78.128.112.139', '45.147.230.242',
       '213.227.154.169', '23.106.215.64', '64.44.135.171', '185.173.34.75', '108.62.118.133', '172.93.193.21', '194.135.24.240',
       '185.217.1.23', '190.123.44.220', '172.93.181.105', '23.82.141.226','23.106.215.123', '23.82.141.117', '172.93.201.141',
       '172.241.29.144']

cs_dataset = pd.DataFrame(columns=columns, dtype=object)
normal_dataset = pd.DataFrame(columns=columns, dtype=object)
test_dataset = pd.DataFrame(columns=columns, dtype=object)

#Create train dataset
for filename in os.listdir(train_folder):
    f = os.path.join(train_folder, filename)
    if os.path.isfile(f):
        path = f.split("/")
        pathroot = path[0]
        x = path[4].split(".")
        l = len(x)
        file = x[0]
        ext = x[l-1]
        if ext == 'DS_Store':
            continue
        d = pd.read_csv(f, header=None, names=columns)
        if file == 'cicids17' or file == 'ctu-normal-20':
            normal_dataset = normal_dataset.append(d)
        else:
            cs_dataset = cs_dataset.append(d)
cs_dataset['label'] = 1
normal_dataset['label'] = 0
dataset = normal_dataset.append(cs_dataset)
dataset = dataset.reset_index(drop=True)
print(dataset)
print('The total amount of normal connections is ' + str(len(normal_dataset)))
print('The total amount of malicious connections is ' + str(len(cs_dataset)))
#dataset.to_csv("Dataset/group-dataset/dataset/group-dataset.csv")

#Create test dataset
for filename in os.listdir(test_folder):
    f = os.path.join(test_folder, filename)
    if os.path.isfile(f):
        x = f.split(".")
        l = len(x)
        ext = x[l-1]
        if ext == 'DS_Store':
            continue
        d = pd.read_csv(f, header=None, names=columns)
        test_dataset = test_dataset.append(d)

test_dataset['label'] = 0
#Set CS records with label = 1
for x in mal:
    test_dataset.loc[test_dataset['dst_ip'] == x, 'label'] = 1

test_dataset = test_dataset.reset_index(drop=True)
print(test_dataset)

n_d = test_dataset[test_dataset['label']==0]
m_d = test_dataset[test_dataset['label']==1]
print('The total amount of normal connections is ' + str(len(n_d)))
print('The total amount of malicious connections is ' + str(len(m_d)))

#test_dataset.to_csv("Dataset/group-dataset/dataset/test-dataset.csv")

'''
#Individual csv file to test
dat = pd.read_csv("Dataset/rand5.txt", header=None)
dat['label'] = 1
print(dat)
dat.to_csv("Dataset/test5.csv")
'''
