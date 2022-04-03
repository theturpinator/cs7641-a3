import dataset
import feature_transformation
import time
import model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
import random as rand

alg = 'PCA'
dataset_name = 'wine-quality'
model_name = 'kmeans'
model_type = 'unsupervized'

dataset_args = {
    "test_size": .3
}

feature_reduction_args = {
    'n_components': 11
}


#PCA

d = dataset.Dataset(dataset_name, **dataset_args)

d.get_dataset(model_type, model_name, dataset_name)

t = feature_transformation.FeatureTransformation(d.X, algorithm=alg, **feature_reduction_args)


#scree plot
if alg == 'PCA':

    y_axis = np.array(t.model.explained_variance_ratio_)
    x_axis = range(1,len(y_axis)+1)
    plt.bar(x_axis, y_axis)
    plt.xticks(x_axis)
    plt.ylabel('Explained Variance %')
    plt.xlabel('Principal Component')
    plt.title('PC Scree Plot')
    plt.show()

    plt.cla()
    plt.clf()

    y_axis = np.array(t.model.explained_variance_ratio_).cumsum()
    x_axis = range(1,len(y_axis)+1)
    plt.bar(x_axis, y_axis)
    plt.xticks(x_axis)
    plt.ylabel('Explained Variance % (Cumulative)')
    plt.xlabel('Principal Component')
    plt.title('PC Scree Plot (Cumulative)')
    plt.show()


#ICA
if alg == 'ICA':
    d = dataset.Dataset(dataset_name, **dataset_args)
    d.get_dataset('unsupervized', 'kmeans', dataset_name)

    pre_ica = d.X.cov().round(2).reset_index(drop=True)
    pre_ica.columns = range(pre_ica.shape[1])

    print('Pre-ICA Covariance Matrix:', pre_ica)

    t = feature_transformation.FeatureTransformation(d.X, algorithm='ICA')
    data = np.array(t.transformed_data)

    print('Post-ICA Covariance Matrix: ', pd.DataFrame(data).cov().round(2))


#RP
if alg == 'RP':
    d = dataset.Dataset(dataset_name, **dataset_args)
    d.get_dataset('unsupervized', 'kmeans', dataset_name)

    st = time.time()
    pca = feature_transformation.FeatureTransformation(d.X, algorithm='PCA')
    print(f'PCA train time: {round(time.time() - st, 3)}')

    st = time.time()
    ica = feature_transformation.FeatureTransformation(d.X, algorithm='ICA')
    print(f'ICA train time: {round(time.time() - st, 2)}')

    st = time.time()
    rp = feature_transformation.FeatureTransformation(d.X, algorithm='RP', **{"n_components": 8})
    print(f'RP train time: {round(time.time() - st, 3)}')


#MAD
if alg == 'MAD':
    plt.bar(t.mda.index, height=t.mda)
    plt.title('MAD Distribution')
    plt.xticks(rotation = 45, fontsize = 7)
    plt.ylabel('MAD')
    plt.show()