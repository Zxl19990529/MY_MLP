from __future__ import division # precise division ex: 3/4 = 0.75
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        return ((target - input) ** 2).mean(axis=0).sum() / 2.

    def backward(self, input, target):
        '''Your codes here'''
        return target - input


class MSE_loss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Mean square error'''

        # return ((target - input) ** 2).mean(axis=0).sum() / 2.
        loss = np.sum((target-input)**2).mean()
        return loss

    def backward(self, input, target):
        # the grad is y - y_hat
        return target - input

def cal_acc(output,label):
    r'''
    Calculate the mean accuracy of the regression result for all samples, whose number equals to batch_size.
    '''
    error = output - label
    error = np.sum(np.abs(error/np.abs(label)),axis = 0)/len(label)
    mean_correct = max(1 - error,0)
    return mean_correct

def cal_classification_acc(output,label):
    correct = np.sum(np.argmax(output, axis=1) == label)
    return correct / len(label)

def onehot_encoding(label, max_num_class):
    r'''
    Convert to classification format
    label: [2,0,1]
    return: [
        [0,0,1]
        [1,0,0]
        [0,1,0]
    ]
    '''
    encoding = np.eye(max_num_class)
    encoding = encoding[label]
    return encoding