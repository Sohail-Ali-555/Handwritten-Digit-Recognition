# Python program for training our ML classifier model(s)
# which will be trained on MNIST dataset.

# step 0: importing necessary modules
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
import pickle

# step 1: fetch mnist data set, one of the latest versions from sklearn module itself
mnist = fetch_openml('mnist_784', parser="auto")

# step 2: Setting up Features (X) & Labels/Targets (Y)
X = mnist['data']
Y = mnist['target']

# we can also get the dimensions (rows, columns) of data set
print("Shape X", X.shape)
print("Shape Y", Y.shape)

# make Numpy array of features (X), which is free of header row
X_np = np.array(X)

# step 3: slicing up datasets both features and labels required for training

"""
In MNIST dataset, the starting first 60000 data records are already
dedicated / categorised for training and later 10000 for testing.
"""
# training features
X_train = X_np[:60000]
# training labels
Y_train = Y[:60000]

# step 4: Shuffling the training datasets for better training / learning
#         and to avoid over-fitting
#         Shuffling will be done through NumPy's random function
shuffle_index = np.random.permutation(60000)

X_train = X_train[shuffle_index]
Y_train = Y_train[shuffle_index]

""" 
Beginning of ML training / learning session
"""
# step 5: Using Logistic Regression for binary classification
#         include tolerance to increase speed, but consumes cpu
#         also include 'lbfgs' solve to increase speed

# list to store classifier models for each of the 10 digits (0-9)
models_list = []

# MAIN LOOP: (training)
# this loop will train a classifier model one at a time for one digit
# each model will be a binary classifier, which will give result as either True or False
# for a given digit
for i in range(10):
    models_solo = LogisticRegression(tol=0.1, solver='lbfgs')

    # can also do without solver
    # models_solo = LogisticRegression(tol=0.1)

    # can also select number of training iterations, higher number of iterations
    # lead to more training time and better model
    # models_solo = LogisticRegression(tol=0.1, solver='lbfgs', max_iter=1000)

    # normally, labels are stored as character or strings, here numbered characters
    # will be converted to integer numeric type
    Y_train = Y_train.astype(np.int8)

    # only those labels will be selected whose numeric int value will be equal to 'i'
    # and 'i' iterates from 0 - 9
    Y_train_digit = (Y_train == i)

    # final model training
    models_solo.fit(X_train, Y_train_digit)

    # add the resultant model to models list
    models_list.append(models_solo)

print("Model Developed")

# step 6: Final store the models-list as a pickled object
#         which can be unpacked later on and used whenever required
file_models = "models.pkl"
file_models_obj = open(file_models, 'wb')
pickle.dump(models_list, file_models_obj)
