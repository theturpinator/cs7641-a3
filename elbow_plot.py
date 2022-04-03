import dataset
import feature_transformation
import time
import model
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
import random as rand

alg = 'PCA'
dataset_name = 'digits'
dataset_args = {
    "test_size": .3
}

model_type = 'em'
model_args = {
    #"init": "random"
    "init_params": "random",
}

feature_transformation_args = {
    "n_components": 21
}

d = dataset.Dataset(dataset_name, **dataset_args)
d.get_dataset('unsupervized', model_type, dataset_name)

t = feature_transformation.FeatureTransformation(d.X, algorithm=alg, **feature_transformation_args).transformed_data

inertia = []

if model_type == 'kmeans':
    min = 1
    max = 25
    for i in range(min,max+1):
        model_args['n_clusters'] = i
        m = model.Model(model_type, **model_args)
        m.model.fit(t)
        inertia.append(m.model.inertia_)
else:
    min = 1
    max = 25
    for i in range(min,max+1):
        model_args['n_components'] = i
        m = model.Model(model_type, **model_args)
        m.model.fit(t)
        inertia.append(m.model.bic(t))

plt.plot(np.arange(min, max+1, step=1), inertia)
plt.title('Elbow Plot')

if model_type == 'kmeans':
    plt.ylabel('Inertia (Variance)')
    plt.xlabel('K Value')
else:
    plt.ylabel('BIC')
    plt.xlabel('Number of Components')
plt.show()









