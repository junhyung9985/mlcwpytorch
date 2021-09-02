import torch
import math
import numpy as np
from skmultilearn import dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
# Fully connected neural network with one hidden layer


class ExternalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, opt = 'train', top = None, scaler = MinMaxScaler, random_state = 42, test_size = 0.25 ):

        path_to_arff_file = '../yahoo/' + dataset_name + '1.arff'
        opt = opt.split('_')
        opt = opt[1]
        if dataset_name == 'Arts':
            label_count = 26
        elif dataset_name == 'Business':
            label_count = 30
        elif dataset_name == 'Science':
            label_count = 40
        else:
            label_count = -1
        label_location = 'end'
        arff_file_is_sparse = False

        X, y = dataset.load_from_arff(
            path_to_arff_file,
            label_count=label_count,
            label_location=label_location,
            load_sparse=arff_file_is_sparse
        )

        X = X.toarray()
        y = y.toarray()

        if top is None:
            example_num_per_label = y.sum(axis=0)
            top = 15

            asc_arg = np.argsort(example_num_per_label)
            des_arg = asc_arg[::-1]
            y = y[:, des_arg[:top]]

        X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=test_size, random_state=random_state)
        if scaler != None:
            scaler = scaler()
            scaler.fit(X_tr)
            X_tr = scaler.transform(X_tr)
            X_ts = scaler.transform(X_ts)

        X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.25, random_state=random_state)
        if (opt == 'train'):
            X = X_tr;
            y = y_tr;
            del (X_ts);
            del (y_ts);
            del (X_val);
            del (y_val)
        elif (opt == 'valid'):
            X = X_val;
            y = y_val;
            del (X_tr);
            del (X_ts);
            del (y_tr);
            del (y_ts)
        else:
            X = X_ts;
            y = y_ts;
            del (X_tr);
            del (y_tr);
            del (X_val);
            del (y_val)

        self.X = X
        self.y = y
        self.length = X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    def __len__(self):
        return self.length


class MultilabelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, opt='train', top = None, scaler = MinMaxScaler, random_state = 42, test_size = 0.25, embed_dir = None):

        opt = opt.split('_')


        X, y, _, _ = dataset.load_dataset(dataset_name, opt[0])
        X, y = X.toarray(), y.toarray()

        self.embed = None;
        if(embed_dir != None):
            embed = pd.read_csv(embed_dir, delimiter=',')
            embed = embed.to_numpy()
            self.embed = torch.from_numpy(embed)

        if top is not None:
            example_num_per_label = y.sum(axis=0)

            asc_arg = np.argsort(example_num_per_label)
            des_arg = asc_arg[::-1]
            y = y[:, des_arg[:top]]

        if(opt[0] == 'undivided') :
            X_tr, X_ts, y_tr, y_ts = train_test_split(X,y, test_size=test_size, random_state=random_state)
            if scaler != None:
                scaler = scaler()
                scaler.fit(X_tr)
                X_tr = scaler.transform(X_tr)
                X_ts = scaler.transform(X_ts)

            X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size = 0.25, random_state = random_state)

            if (opt[1] == 'train'):
                X = X_tr; y = y_tr; del(X_ts); del(y_ts) ; del(X_val); del(y_val)
            elif(opt[1] == 'valid'):
                X = X_val; y = y_val ; del(X_tr); del(X_ts); del(y_tr); del(y_ts)
            else:
                X = X_ts; y = y_ts; del(X_tr); del(y_tr); del(X_val); del(y_val)
        else :
            X_tr, y_tr, _, _ = dataset.load_dataset(dataset_name, 'train')
            X_tr = X_tr.toarray()

            if scaler != None:
                scaler = scaler()
                scaler.fit(X_tr)
                X = scaler.transform(X)
                del(X_tr)

            if top is not None:
                y_tr = y_tr.toarray()
                sum = y_tr.sum(axis=0)

                top_label_index = []

                for i in range(top):
                    largest_index = np.argmax(sum)
                    top_label_index.append(largest_index)
                    sum = np.delete(sum, largest_index)

                y = y[:, top_label_index]

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

        self.length = self.X.shape[0]

    def __getitem__(self, index):
        if(self.embed!= None) : return self.X[index], self.y[index], self.embed[index]
        return self.X[index], self.y[index]

    def __len__(self):
        return self.length