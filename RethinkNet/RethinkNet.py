import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import math
import copy
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics as metrics
from skmultilearn import dataset


######################## RethinkNet ########################

def arch_001(input_size, output_size, dropout=0.25, activation=nn.Sigmoid, rnn_unit='lstm'):
    embed_size = 128

    input_layer = nn.Linear(input_size, embed_size)
    input_size = embed_size

    if rnn_unit == 'rnn':
        rnn_unit = nn.RNN(input_size, embed_size, 1)  # , dropout = dropout )
    elif rnn_unit == 'lstm':
        rnn_unit = nn.LSTM(input_size, embed_size, 1)  # , dropout = dropout)
    elif rnn_unit == 'gru':
        rnn_unit = nn.GRU(input_size, embed_size, 1)  # , dropout = dropout)
    else:
        NotImplementedError()

    RNN = rnn_unit

    dec = nn.Sequential(
        nn.Linear(embed_size, output_size),
        activation()
    )
    return input_layer, RNN, dec, embed_size


class RethinkNet(nn.Module):
    def __init__(self, input_size, output_size, architecture="arch_001", rethink_time=2, rnn_unit='lstm',
                 reweight='None', device='cpu'):
        super(RethinkNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.rnn_unit = rnn_unit
        self.input_layer, self.rnn, self.dec, self.embed_size = globals()[architecture](self.input_size,
                                                                                        self.output_size,
                                                                                        rnn_unit=rnn_unit)
        self.b = rethink_time + 1
        self.reweight = reweight
        self.device = device

    def prep_Y(self, Y):
        return torch.cat([Y for _ in range(self.b)], axis=0)

    def prep_X(self, X):
        return X.view(1, X.shape[0], -1)

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.embed_size).to(self.device).double(), torch.zeros(1, batch_size,
                                                                                                 self.embed_size).to(
            self.device).double()

    def predict_proba(self, X):
        hist = [0 for _ in range(self.b)]

        h_0, c_0 = self.init_hidden(X.shape[0])
        hidden = (h_0, c_0)
        if (self.rnn_unit == 'rnn'): hidden = h_0

        X_embed = self.input_layer(X)
        X_embed = self.prep_X(X_embed)

        for i in range(self.b):
            embed, hidden = self.rnn(X_embed, hidden)
            out = self.dec(torch.squeeze(embed))
            hist[i] = out

        return hist

    def predict(self, X):
        hist = self.predict_proba(X)
        hist = [(i > Variable(torch.Tensor([0.5]).to(DEVICE))).double() * 1 for i in hist]
        return hist

    def forward(self, X):
        output = [0 for _ in range(self.b)]

        h_0, c_0 = self.init_hidden(X.shape[0])
        hidden = (h_0, c_0)
        if (self.rnn_unit == 'rnn'): hidden = h_0

        X_embed = self.input_layer(X)
        X_embed = self.prep_X(X_embed)

        for i in range(self.b):
            embed, hidden = self.rnn(X_embed, hidden)
            out = self.dec(torch.squeeze(embed))
            output[i] = out
        output = torch.cat(output, axis=0)
        # for prediction
        return output


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # torch.device('cpu')
    # device = 'cpu'
    print("Example")
    from Utils import Nadam, log_likelihood_loss, jaccard_score, MultilabelDataset

    train_dataset = MultilabelDataset(dataset_name='scene', opt='undivided_train', random_state=7)
    test_dataset = MultilabelDataset(dataset_name='scene', opt='undivided_test', random_state=7)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=128,
                                               shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=test_dataset.X.shape[0],
                                              shuffle=False, num_workers=0)
    model = RethinkNet(train_dataset.X.shape[1], train_dataset.y.shape[1], device=device).to(device).double()
    criterion = nn.BCELoss()
    prediction = np.expand_dims(np.zeros(test_loader.dataset.y.shape[1]), axis=0)
    y_true = np.expand_dims(np.zeros(test_loader.dataset.y.shape[1]), axis=0)

    optimizer = Nadam([{'params': model.rnn.parameters(), 'weight_decay': 1e-05}, {'params': model.dec.parameters()}])
    for _ in range(1000):
        for X, labels in train_loader:
            optimizer.zero_grad()
            X = X.to(device).double()
            labels = model.prep_Y(labels)
            labels = labels.to(device).double()

            output = model(X)
            ls = criterion(output, labels)
            ls.backward()
            optimizer.step()

    with torch.no_grad():
        for X, labels in test_loader:
            X = X.to(device).double()
            labels = labels.to(device).double()
            outputs = model.predict_proba(X)
            predicted = torch.squeeze(outputs[-1])
            frac_labels = (labels.cpu()).numpy()
            y_true = np.concatenate((y_true, frac_labels), axis=0)

            frac_prediction = (predicted.cpu()).numpy()
            prediction = np.concatenate((prediction, frac_prediction), axis=0)
        prediction = np.delete(prediction, 0, 0)
        prediction_proba = prediction.copy()

        y_true = np.delete(y_true, 0, 0)
        for i in range(prediction.shape[0]):
            is_correct = 0
            for j in range(prediction.shape[1]):
                if (prediction[i, j] >= 0.5):
                    prediction[i, j] = 1
                elif (prediction[i, j] < 0.5):
                    prediction[i, j] = 0

        import sklearn.metrics as metrics

        f1_micro_score = metrics.f1_score(y_true, prediction, average='micro')
        f1_macro_score = metrics.f1_score(y_true, prediction, average='macro')
        accuracy = metrics.accuracy_score(y_true, prediction)
        cll_loss = log_likelihood_loss(y_true, prediction_proba)
        jaccard = jaccard_score(y_true, prediction)
        hamming_score = 1 - metrics.hamming_loss(y_true, prediction)
        print('ema = %.5f' % (accuracy))
        print('f1_micro_score = %.5f' % (f1_micro_score))
        print('f1_macro_score = %.5f' % (f1_macro_score))
        print('cll_loss = %.5f' % (cll_loss))

'''
    To Implement
        - Loss function (Reweighted CSMLC)
        - Recurrent Dropout???
'''