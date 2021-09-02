import torch
import math
import numpy as np
from skmultilearn import dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
# Fully connected neural network with one hidden layer
############################# UTILS #############################


EPSILON = 10e-18

def log_likelihood_loss(y, y_prob):
    log_likelihood_loss = 0
    for row in range(y.shape[0]):
        if y.ndim == 1:
            if y[row] == 1:
                log_likelihood_loss += (-math.log(np.max(EPSILON, y_prob[row])))
            else:
                log_likelihood_loss += (-math.log(1 - np.min(1 - EPSILON, y_prob[row])))
        elif y.ndim == 2:
            for col in range(y.shape[1]):
                if y[row, col] == 1:
                    log_likelihood_loss += (-math.log(EPSILON if y_prob[row, col] == 0 else y_prob[row, col]))
                else:
                    log_likelihood_loss += (-math.log(1 - (EPSILON if y_prob[row, col] == 1 else y_prob[row, col])))
    return log_likelihood_loss


def jaccard_score(y, y_pred):
    y_ = y.tolist()
    y_pred = y_pred.tolist()
    jaccard_dis = 0
    all_zero  = 0
    for row in range(y.shape[0]):
        denom = [int(i)| int(j) for i, j in zip(y_[row], y_pred[row])]
        nom = [int(i) & int(j) for i, j in zip(y_[row], y_pred[row])]
        if(sum(denom) == 0): all_zero += 1
        else : jaccard_dis += sum(nom)/ sum(denom)
    jaccard_dis /= (y.shape[0] - all_zero)
    return jaccard_dis
