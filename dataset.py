from cProfile import label
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_diabetes, load_digits, load_wine, load_breast_cancer
import seaborn as sb
import matplotlib.pyplot as plt

class Dataset:

    def __init__(self, dataset_name, **kwargs):
        self.dataset_name = dataset_name
        self.kwargs = kwargs

    def get_dataset(self, problem_type, model_type, name):

        match name:
            
            case 'wine-quality':
                path = 'data/WineQT.csv'
                label_name = 'quality'
                columns_to_drop_from_X = [label_name, 'Id']

                # read dataset into pandas dataframe
                df = pd.read_csv(path)


                # split into attributes and labels, convert to NumPy
                X = df.drop(columns_to_drop_from_X, axis=1)

                df.loc[df[label_name] < 5, 'new_label'] = 'bad'
                df.loc[(df[label_name] >= 5) & (df[label_name] <= 6), 'new_label'] = 'avg'
                df.loc[df[label_name] > 6, 'new_label'] = 'good'

                y = df['new_label']

            case 'iris':
                X, y = load_iris(return_X_y=True, as_frame=True)

            case 'diabetes':
                X, y = load_diabetes(return_X_y=True, as_frame=True)

            case 'digits':
                X, y = load_digits(return_X_y=True, as_frame=True)

            case 'wine-region':
                X, y = load_wine(return_X_y=True, as_frame=True)

            case 'breast-cancer':
                X, y = load_breast_cancer(return_X_y=True, as_frame=True)


        #normalize data
        if name != 'digits':
            X = (X-X.mean())/X.std() #reference: https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame

        self.X = X
        self.y = y

        return X, y

        


    def print_sample(self):
        print('attributes', self.X.head())
        print('labels', self.y.head())

    def pairplot(self, classes, model):


        if len(classes) == self.X_train.shape[0] and model == 'kmeans' or model == 'em':

            self.X_train['class'] = pd.Series(classes)

            sb.pairplot(self.X_train, hue='class')

        elif self.dataset_name == 'wine-quality' and len(classes) == self.X_test.shape[0]:

            self.X_test['class'] = pd.Series(classes)

            sb.pairplot(self.X_test, hue='class')


        plt.show()





        

        