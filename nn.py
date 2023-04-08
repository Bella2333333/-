from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np

import random

class TwoLayerNet(object):

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):

        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y = None, reg=0.0, if_test = False):

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        H = np.maximum(X.dot(W1) + b1, 0)
        scores = H.dot(W2) + b2

        if y is None:
            return scores

        # 计算loss
        loss = 0.0

        row_max  = np.max(scores,1)
        get_err = np.exp(scores.T - row_max).T
        get_p = (get_err.T / np.sum(get_err,1)).T
        loss += -np.sum(np.log(get_p[range(N),y]))
        
        # 平均loss
        loss /= N
        if if_test:
            return loss 
        loss += reg * np.sum(W1 * W1) + reg * np.sum(W2 * W2) 

        # 反向传播计算梯度
        grads = {}

        d_scores = get_p.copy()
        d_scores[range(N), list(y)] -= 1 # 每个分量求导
        d_scores /= N
        grads['W2'] = H.T.dot(d_scores) + reg * W2
        grads['b2'] = np.sum(d_scores, axis = 0)
    
        dh = d_scores.dot(W2.T)
        dh_ReLu = (H > 0) * dh
        grads['W1'] = X.T.dot(dh_ReLu) + reg * W1
        grads['b1'] = np.sum(dh_ReLu, axis = 0)

        return loss, grads

    def train(self, X, y, X_val, y_val, learning_rate = 1e-3, learning_rate_decay = 0.95,
              reg = 5e-6, num_iters = 100, batch_size = 100, verbose = False):
        
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # 优化器SGD
        train_loss_hist = []
        val_loss_hist = []
        train_acc_hist = []
        val_acc_hist = []
        index = list(range(num_train))
        
        for iter in range(num_iters):
            if iter % iterations_per_epoch == 0:
                random.shuffle(index)
            X_batch = None
            y_batch = None

            ii_ = iter % iterations_per_epoch
            it_index = index[int(ii_* batch_size):int((ii_+1)* batch_size)]
            X_batch = X[it_index]
            y_batch = y[it_index]

            # 计算loss和梯度
            train_loss, grads = self.loss(X_batch, y = y_batch, reg = reg)
            val_loss = self.loss(X_val, y = y_val, reg = reg, if_test = True)
            train_loss_hist.append(train_loss)
            val_loss_hist.append(val_loss)

            self.params['W2'] += - learning_rate * grads['W2']
            self.params['b2'] += - learning_rate * grads['b2']
            self.params['W1'] += - learning_rate * grads['W1']
            self.params['b1'] += - learning_rate * grads['b1']

            if verbose and iter % 100 == 0:
                print('第 %d 次迭代: train loss %f validation loss %f' % (iter, train_loss, val_loss))

            # 计算accuracy
            train_acc = (self.predict(X_batch) == y_batch).mean()
            val_acc = (self.predict(X_val) == y_val).mean()
            train_acc_hist.append(train_acc)
            val_acc_hist.append(val_acc)

            # 对每个epoch, 检查train和val的accuracy，改变学习率
            if iter % iterations_per_epoch == 0:
                learning_rate *= learning_rate_decay

        return {
          'train_loss_hist': train_loss_hist,
          'val_loss_hist': val_loss_hist,
          'train_acc_hist': train_acc_hist,
          'val_acc_hist': val_acc_hist,
        }

    def predict(self, X):

        h = np.maximum(0, X.dot(self.params['W1']) + self.params['b1'])
        scores = h.dot(self.params['W2']) + self.params['b2']
        y_pred = np.argmax(scores, axis=1)

        return y_pred
    
    def dropout(self, p):
        retain_prob = 1. - p        
        W1, W2 = self.params['W1'], self.params['W2']
        sample1 = np.random.binomial(n = 1, p = retain_prob, size = W1.shape)
        sample2 = np.random.binomial(n = 1, p = retain_prob, size = W2.shape)
        W1 *= sample1
        W2 *= sample2
        self.params['W1'], self.params['W2'] = W1, W2

    def save_model(self,model= 'bestmodelparams.npz'):
        np.savez(model,w1 = self.params['W1'], w2 = self.params['W2'], b1 = self.params['b1'], b2 = self.params['b2'])
    
    def load_model(self,model = 'bestmodelparams.npz'):
        p = np.load(model)
        self.params['W1'] = p['w1']
        self.params['W2'] = p['w2']
        self.params['b1'] = p['b1']
        self.params['b2'] = p['b2']




  
        
