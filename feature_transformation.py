import sklearn as skl
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.random_projection import GaussianRandomProjection as GRP
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

class FeatureTransformation:

    def __init__(self, data, algorithm='PCA', **kwargs):
        self.algorithm = algorithm
        self.transformed_data = self.transform(data, algorithm, **kwargs)

    def transform(self, data, alg, **kwargs):

        match alg:

            case 'PCA':
                model = PCA(**kwargs)

            case 'ICA':
                model = FastICA(**kwargs)

            case 'RP':
                model = GRP(**kwargs)

            case 'MAD':
                model = None


        self.model = model
        
        if alg == 'MAD':

            n_components = kwargs['n_components']

            st = time.time()
            #ref: https://www.analyticsvidhya.com/blog/2020/10/feature-selection-techniques-in-machine-learning/
            mda = (data - data.mean(axis=0)).abs().sum(axis=0)/data.shape[0]
            self.mda = mda

            mda_ordered = mda.sort_values(axis=0, ascending=False)
            
            selected_data = data[mda_ordered.iloc[:n_components].index]
            selected_data.columns = list(range(n_components))
            
            print(f'Feature Transformation Time: {round(time.time() - st, 2)}ms')

            return selected_data


        st = time.time()
        transformed_features = model.fit_transform(data)
        print(f'Feature Transformation Time: {time.time() - st}ms')

        return pd.DataFrame(transformed_features)

    