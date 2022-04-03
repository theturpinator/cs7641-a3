from re import X
import dataset
import time
import model
import feature_transformation
from sklearn.metrics import accuracy_score, silhouette_score, davies_bouldin_score, mutual_info_score
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from scipy.stats import norm
import statistics
import numpy as np


# Define Params
# ----------------
model_type = 'kmeans'
feature_reduction_type = 'PCA'
dataset_name = 'digits'

model_args = {

    #k-means
    "n_clusters": 7,
    "init": "random"

    #em
    #"init_params": "random",
    #"n_components": 4,
    #"covariance_type": "full",
    #"n_init": 1

}

feature_reduction_args = {
    "n_components": 7
}

dataset_args = {
    "test_size": .3
}

cv_folds = 5
# ----------------


# Create model and dataset
m = model.Model(model_type, **model_args)
d = dataset.Dataset(dataset_name, **dataset_args)


# split data into test/train
X, y = d.get_dataset(m.problem_type, model_type, dataset_name)

if m.problem_type == 'unsupervized':
    X_train = feature_transformation.FeatureTransformation(X, feature_reduction_type, **feature_reduction_args).transformed_data
    y_train = y

# perform cross validation
if m.problem_type == 'supervized':

    X_train, X_test, y_train, y_test = train_test_split(X, y, **dataset_args, random_state=100, stratify=y)

    cross_val_scores = cross_val_score(m.model, X_train, y_train, cv=cv_folds)
    m.compile_stats(X_train, y_train, cv_folds)


# fit model
if m.problem_type == 'supervized':
    train_st = time.time()
    m.model.fit(X_train, y_train)
    train_time = time.time() - train_st

else:
    train_st = time.time()
    m.model.fit(X_train)
    train_time = time.time() - train_st


# get prediction
if m.problem_type == 'supervized':
    X_pred = X_test
else:
    X_pred = X_train

predict_st = time.time()
pred = m.model.predict(X_pred)
predict_time = time.time() - predict_st

if m.problem_type == 'supervized':
    accuracy = accuracy_score(y_test, pred)
    class_accuracy = m.get_class_accuracy(y_test, pred)


#print high level stats
print('-----------------------------')
print('Model: ', model_type)
print('Dataset: ', dataset_name)
print('Feature Transformation Algorithm: ', feature_reduction_type)
print('model params: ', model_args)
print('-------RESULTS-------')
if m.problem_type == 'supervized':
    print('Test Accuracy: ', round(accuracy*100, 2), '%')
else:
    print('Silhouette Score: ', silhouette_score(X_train, pred))
    print('Davies-Bouldin Score', davies_bouldin_score(X_train, pred))
print('Train time: ', round((train_time*1000), 2), 'ms')
print('Predict time:', round((predict_time*1000), 2), 'ms')
print('-----------------------------')


#print unsupervized metrics
if m.problem_type == 'unsupervized':


    # print two dimensions of the transformed data
    X_train = pd.DataFrame(X_train)
    X_train['class'] = pd.Series(pred)

    c1 = 0
    c2 = 1

    plt.scatter(X_train[c1].where(X_train["class"] == 0), X_train[c2].where(X_train["class"] == 0), color="red")
    plt.scatter(X_train[c1].where(X_train["class"] == 1), X_train[c2].where(X_train["class"] == 1), color="blue")
    plt.scatter(X_train[c1].where(X_train["class"] == 2), X_train[c2].where(X_train["class"] == 2), color="green")
    plt.scatter(X_train[c1].where(X_train["class"] == 3), X_train[c2].where(X_train["class"] == 3), color="violet")
    plt.scatter(X_train[c1].where(X_train["class"] == 4), X_train[c2].where(X_train["class"] == 4), color="orange")
    plt.scatter(X_train[c1].where(X_train["class"] == 5), X_train[c2].where(X_train["class"] == 5), color="black")
    plt.scatter(X_train[c1].where(X_train["class"] == 6), X_train[c2].where(X_train["class"] == 6), color="teal")
    plt.scatter(X_train[c1].where(X_train["class"] == 7), X_train[c2].where(X_train["class"] == 7), color="gray")
    plt.scatter(X_train[c1].where(X_train["class"] == 8), X_train[c2].where(X_train["class"] == 8), color="yellow")
    plt.scatter(X_train[c1].where(X_train["class"] == 9), X_train[c2].where(X_train["class"] == 9), color="purple")
    plt.xlabel(f'{feature_reduction_type} {c1}')
    plt.ylabel(f'{feature_reduction_type} {c2}')
    
    #sb.pairplot(X_train, hue='class', height=1.1)
    plt.show()





    # see breakdown of targets within each class
    y_train = pd.DataFrame(y_train)
    y_train["class"] = pd.Series(pred)

    if 'target' in y_train.columns:
        cols = ['class', 'target']
    else:
        cols = ['class', 'new_label']

    grouped_data = y_train.groupby(cols).size()
    #print(grouped_data)




    #Mutual information
    print(f'Mutual Information Score: {round(mutual_info_score(X_train[c1], X_train[c2]), 2)}')





    #Plot data distribution
    nd_data = X_train[c1].sort_values()

    #ref: https://www.geeksforgeeks.org/how-to-plot-a-normal-distribution-with-matplotlib-in-python/
    mean = statistics.mean(nd_data)
    sd = statistics.stdev(nd_data)

    plt.cla()
    plt.clf()

    plt.plot(nd_data, norm.pdf(nd_data, mean, sd))
    plt.title('ICA Compatibility - Gaussian Test')
    plt.show()