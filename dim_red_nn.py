from re import X
import dataset
import time
import model
import feature_transformation
from sklearn.metrics import accuracy_score, silhouette_score, davies_bouldin_score, mutual_info_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from scipy.stats import norm
import statistics
import numpy as np

# Define Params
# ----------------
model_type = 'nn'
feature_reduction_type = 'PCA'
dataset_name = 'wine-quality'

model_args = {

    #k-means
    #"n_clusters": 7,
    #"init": "random"

    #em
    #"init_params": "random",
    #"n_components": 4,
    #"covariance_type": "full",
    #"n_init": 1

    #nn params
    "hidden_layer_sizes": (100, 100),
    "solver": "adam",
    "random_state": 100,
    "learning_rate": "constant",
    "learning_rate_init": 1e-2,
    "max_iter": 1000
}

feature_reduction_args = {
    "n_components": 5
}

dataset_args = {
    "test_size": .3
}

cv_folds = 5
# ----------------

# Create model and dataset
m = model.Model(model_type, **model_args)
d = dataset.Dataset(dataset_name, **dataset_args)


# get dataset and normalize
X, y = d.get_dataset(m.problem_type, model_type, dataset_name)


#apply feature transformation
X_transformed = feature_transformation.FeatureTransformation(X, feature_reduction_type, **feature_reduction_args).transformed_data


#split into train/test
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, **dataset_args, random_state=100, stratify=y)

#cross validation
cross_val_scores = cross_val_score(m.model, X_train, y_train, cv=cv_folds)
m.compile_stats(X_train, y_train, cv_folds)



#fit model to training data
st = time.time()
m.model.fit(X_train, y_train)
train_time = time.time() - st

#make prediction
st = time.time()
pred = m.model.predict(X_test)
predict_time = time.time() - st

#determine accuracy
accuracy = accuracy_score(y_test, pred)
class_accuracy = m.get_class_accuracy(y_test, pred)

#print stats
print('-----------------------------')
print('Model: ', model_type)
print('Dataset: ', dataset_name)
print('Feature Transformation Algorithm: ', feature_reduction_type)
print('model params: ', model_args)
print('-------RESULTS-------')
print('Test Accuracy: ', round(accuracy*100, 2), '%')
print('Train time: ', round((train_time*1000), 2), 'ms')
print('Predict time:', round((predict_time*1000), 2), 'ms')
print('Num Iterations:', m.model.n_iter_)
print('-----------------------------')