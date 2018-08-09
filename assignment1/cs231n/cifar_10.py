

from __future__ import print_function
from six.moves import cPickle as pickle
import numpy as np
import os
# from scipy.misc import imread
import platform

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in range(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      # distances = np.sqrt(np.sum(np.square(self.Xtr - X[i, :]), axis=1))
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example
      # print(min_index)

    return Ypred


def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = load_pickle(f)
    # print(datadict)
    n = 2000
    X = datadict['data'][0:n]
    Y = datadict['labels'][0:n]
    # print(X.shape)
    # print("================")
    # print(X)
    # print(X.reshape(10000, 3, 32, 32)[0, :, :, :])
    # print("================")
    # print(X.reshape(10000, 3, 32, 32).transpose(0,2,3,1)[0, :, :, :])
    X = X.reshape(n, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    # print(Y)
    return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,2):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte

Xtr, Ytr, Xte, Yte = load_CIFAR10('datasets/cifar-10-batches-py/')
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072
# print(Xtr_rows)

# =======================================================================
# nn = NearestNeighbor() # create a Nearest Neighbor classifier class
# nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
# Yte_predict = nn.predict(Xte_rows) # predict labels on the test images
# # and now print the classification accuracy, which is the average number
# # of examples that are correctly predicted (i.e. label matches)
# print(Yte)
# print("==========")
# print(Yte_predict)
# print('accuracy: %f' % ( np.mean(Yte_predict == Yte) ))

# ======================================================================= validation set
# assume we have Xtr_rows, Ytr, Xte_rows, Yte as before
# recall Xtr_rows is 50,000 x 3072 matrix
m = 200
Xval_rows = Xtr_rows[:m, :]  # take first 1000 for validation
Yval = Ytr[:m]
Xtr_rows = Xtr_rows[m:, :]  # keep last 49,000 for train
Ytr = Ytr[m:]
# find hyperparameters that work best on the validation set
validation_accuracies = []
for k in [1, 3, 5, 10, 20, 50, 100]:
    # use a particular value of k and evaluation on validation data
    nn = NearestNeighbor()
    nn.train(Xtr_rows, Ytr)
    # here we assume a modified NearestNeighbor class that can take a k as input
    Yval_predict = nn.predict(Xval_rows, k=k)
    acc = np.mean(Yval_predict == Yval)
    print("accuracy: %f" % (acc,))

    # keep track of what works on the validation set
    validation_accuracies.append((k, acc))