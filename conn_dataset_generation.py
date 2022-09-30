import os
import pandas as pd

columns = ['ts', 'uid', 'id.orig_h','id.orig_p','id.resp_h', 'id.resp_p', 'proto', 'service', 'duration', 'orig_bytes',
           'resp_bytes', 'conn_state', 'local_orig', 'local_resp', 'missed_bytes', 'history', 'orig_pkts', 'orig_ip_bytes',
           'resp_pkts', 'resp_ip_bytes', 'tunnel_parents']
columns_drop = ['ts', 'uid', 'local_orig', 'local_resp', 'missed_bytes', 'tunnel_parents']

mal = ['172.93.179.196', '52.18.235.51',  '217.79.243.147', '194.37.97.139', '179.60.146.16', '23.82.141.136', '179.60.150.125', '108.177.235.172', '172.93.201.12', '23.106.215.100',
       '172.241.27.237', '139.60.161.45', '139.60.160.8',  '5.149.249.187', '103.207.42.11', '78.128.112.139', '45.147.230.242',
       '213.227.154.169', '23.106.215.64', '64.44.135.171', '185.173.34.75', '108.62.118.133', '172.93.193.21', '194.135.24.240',
       '185.217.1.23', '190.123.44.220', '172.93.181.105', '23.82.141.226','23.106.215.123', '23.82.141.117', '172.93.201.141',
       '172.241.29.144', '144.202.49.189']

train_folder = 'Dataset/conn-dataset/train'
test_folder = 'Dataset/conn-dataset/test'

cs_dataset = pd.DataFrame(columns=columns, dtype=object)
normal_dataset = pd.DataFrame(columns=columns, dtype=object)
test_dataset = pd.DataFrame(columns=columns, dtype=object)

#Create train dataset
for filename in os.listdir(train_folder):
    f = os.path.join(train_folder, filename)
    if os.path.isfile(f):
        path = f.split("/")
        pathroot = path[0]
        x = path[3].split(".")
        file = x[0]
        ext = x[1]
        if ext == 'DS_Store':
            continue
        d = pd.read_csv(f, sep='\t', header=None, names=columns)
        d = d.drop([0, 1, 2, 3, 4, 5, 6, 7, (len(d) - 1)])
        if file == 'cicids17-normal' or file == 'ctu-normal-20':
            normal_dataset = normal_dataset.append(d, sort=False)
        else:
            cs_dataset = cs_dataset.append(d, sort=False)

cs_dataset['label'] = 1
cs_dataset = cs_dataset[(cs_dataset.service == 'http') | (cs_dataset.service == 'ssl')]
normal_dataset['label'] = 0
dataset = normal_dataset.append(cs_dataset, sort=False)
dataset = dataset.drop(columns=columns_drop)
dataset = dataset.reset_index(drop=True)
print(dataset)
print('The total amount of normal connections is ' + str(len(normal_dataset)))
print('The total amount of malicious connections is ' + str(len(cs_dataset)))
http_d = cs_dataset[cs_dataset['service']=='ssl']
print('Amount of HTTPS Beacon conn is ' + str(len(http_d)))

#dataset.to_csv("Dataset/conn-dataset/dataset.csv")
#cs_dataset.to_csv("Dataset/conn-dataset/dataset/mal_dataset.csv")
#dataset.to_csv("Dataset/conn-dataset/dataset/norm_dataset.csv")

#Create test dataset
for filename in os.listdir(test_folder):
    f = os.path.join(test_folder, filename)
    if os.path.isfile(f):
        x = f.split(".")
        ext = x[1]
        if ext == 'DS_Store':
            continue
        d = pd.read_csv(f, sep='\t', header=None, names=columns)
        d = d.drop([0, 1, 2, 3, 4, 5, 6, 7, (len(d) - 1)])
        test_dataset = test_dataset.append(d)

test_dataset['label'] = 0
#Set CS records with label = 1
for x in mal:
    test_dataset.loc[test_dataset['id.resp_h'] == x, 'label'] = 1

test_dataset = test_dataset.drop(columns=columns_drop)
test_dataset = test_dataset.reset_index(drop=True)
print(test_dataset)
n_d = test_dataset[test_dataset['label']==0]
m_d = test_dataset[test_dataset['label']==1]
print('The total amount of normal connections is ' + str(len(n_d)))
print('The total amount of malicious connections is ' + str(len(m_d)))
https_d = m_d[m_d['service']=='ssl']
print('Amount of HTTPS Beacon conn is ' + str(len(https_d)))
test_dataset.to_csv("Dataset/conn-dataset/test-dataset.csv")


'''
#Individual csv file to test
dat = pd.read_csv("Dataset/conn-dataset/train/amazon.log", sep='\t',header=None, names=columns)
dat = dat.drop([0,1,2,3,4,5,6,7,(len(dat)-1)])
dat = dat.drop(columns=columns_drop)
dat = dat.reset_index(drop=True)
print(dat.to_string())
dat.to_csv("Dataset/test.csv")
'''