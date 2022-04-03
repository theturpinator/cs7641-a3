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
dataset_name = 'wine-quality'

cluster_alg = 'kmeans'
cluster_args = {

    #em
    #"init_params": "random",
    #"n_components": 4,
    #"covariance_type": "full",
    #"n_init": 1

    #k-means
    "n_clusters": 7,
    "init": "random"

}

nn_args = {

    #nn params
    "hidden_layer_sizes": (100, 100),
    "solver": "adam",
    "random_state": 100,
    "learning_rate": "constant",
    "learning_rate_init": 1e-2,
    "max_iter": 1000
}

feature_reduction_type = 'PCA'

feature_reduction_args = {
    "n_components": 7
}


dataset_args = {
    "test_size": .3
}

cv_folds = 5
# ----------------


# Create model and dataset
cluster_model = model.Model(cluster_alg, **cluster_args)
d = dataset.Dataset(dataset_name, **dataset_args)


# get dataset and normalize
X, y = d.get_dataset(cluster_model.problem_type, 'kmeans', dataset_name)

X = feature_transformation.FeatureTransformation(X, feature_reduction_type, **feature_reduction_args).transformed_data

clusters = cluster_model.model.fit_predict(X)

if cluster_alg == 'kmeans':
    X['cluster'] = pd.Series(clusters)
elif cluster_alg == 'em':

    probs = cluster_model.model.predict_proba(X)
    #create a feature for each component, containing the probability that a particular feature belongs to each component
    for i in range(cluster_model.model.n_components):
        key = f'cluster_{i}'
        X[key] = pd.Series(probs[:,i])

nn_model = model.Model('nn', **nn_args)

#split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, **dataset_args, random_state=100, stratify=y)

#cross validation
cross_val_scores = cross_val_score(nn_model.model, X_train, y_train, cv=cv_folds)
nn_model.compile_stats(X_train, y_train, cv_folds)


#fit model to training data
st = time.time()
nn_model.model.fit(X_train, y_train)
train_time = time.time() - st

#make prediction
st = time.time()
pred = nn_model.model.predict(X_test)
predict_time = time.time() - st

#determine accuracy
accuracy = accuracy_score(y_test, pred)
class_accuracy = nn_model.get_class_accuracy(y_test, pred)

#print stats
print('-----------------------------')
print('Model: ', 'Neural Network')
print('Dataset: ', dataset_name)
print('Feature Transformation Algorithm: ', 'K-Means')
print('model params: ', nn_args)
print('-------RESULTS-------')
print('Test Accuracy: ', round(accuracy*100, 2), '%')
print('Train time: ', round((train_time*1000), 2), 'ms')
print('Predict time:', round((predict_time*1000), 2), 'ms')
print('Num Iterations:', nn_model.model.n_iter_)
print('-----------------------------')