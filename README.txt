# cs7641-a3
Please install all required dependencies: Python 3.10, Scikit-learn, Numpy, Pandas, SciPy, Seaborn, and Matplotlib
0. Clone the following repository: https://github.com/theturpinator/cs7641-a3

# FOR ELBOW PLOTS
1. Open elbow_plot.py
2. Specify the type of feature selection algorithm you'd like to run by providing the variable 'alg' a value of 'PCA', 'ICA', 'RP', or 'MAD'
3.Specify the dataset you'd like to use by assigning the dataset_name variables a value of "wine-quality" or "digits"
4. Specify the clustering algorithm you'd like to use by providing the variable model_type a value of 'em' or 'kmeans'
5. Fill out the model_args parameter to specify hyperparameter values into the models. Examples are provided in comments within the dictionary.
6. Fill out the feature_transformation_args variable with any parameters you'd like to provide (namely the number of components ('n_components')
7. Run the program by typing "python elbow_plot.py" in the console

# For Scree Plot (PCA), Covariance Matrices (ICA), MAD Distribution, or runtime comparison (RP)
1. Open dim_red_test.py
2. Specify the type of feature selection algorithm you'd like to run by providing the variable 'alg' a value of 'PCA', 'ICA', 'RP', or 'MAD'
3.Specify the dataset you'd like to use by assigning the dataset_name variables a value of "wine-quality" or "digits"
4. Fill out the feature_reduction_args variable with any parameters you'd like to provide (namely the number of components ('n_components')
5. Run the program by typing "python dim_red_test.py" in the console

# For Dimensionality Transformation/Reduction performance on Neural Network
1. Open dim_red_nn.py
2. Specify the type of feature selection algorithm you'd like to run by providing the variable 'feature_reduction_type' a value of 'PCA', 'ICA', 'RP', or 'MAD'
3.Specify the dataset you'd like to use by assigning the dataset_name variables a value of "wine-quality" or "digits"
4. Fill out the model_args parameter to specify hyperparameter values for the models. (Or leave as the default values)
5. Fill out the feature_reduction_args variable with any parameters you'd like to provide (namely the number of components ('n_components')
6. Run the program by typing "python dim_red_nn.py" in the console

# For adding cluster outputs to our transformed dataset prior to training on a Neural Network
1. Open clustering_dim_red.py
2. Specify the type of feature selection algorithm you'd like to run by providing the variable 'feature_reduction_type' a value of 'PCA', 'ICA', 'RP', or 'MAD'
3.Specify the dataset you'd like to use by assigning the dataset_name variables a value of "wine-quality" or "digits"
4. Specify the clustering algorithm you'd like to use by providing the variable cluster_alg a value of 'em' or 'kmeans'
5. Fill out the cluster_args parameter to specify hyperparameter values into the clustering model. Examples are provided in comments within the dictionary.
6. Fill out the feature_reduction_args variable with any parameters you'd like to provide (namely the number of components ('n_components')
7. Neural Net hyper parameter values can be modified in 'nn_args', if desired
8. Run the program by typing "python clustering_dim_red.py" in the console

# For Feature Transformation/Dataset/Clustering Model comparison
1. Open main.py
2. Specify the type of feature selection algorithm you'd like to run by providing the variable 'feature_reduction_type' a value of 'PCA', 'ICA', 'RP', or 'MAD'
3.Specify the dataset you'd like to use by assigning the dataset_name variables a value of "wine-quality" or "digits"
4. Specify the clustering algorithm you'd like to use by providing the variable model_type a value of 'em' or 'kmeans'
5. Fill out the model_args parameter to specify hyperparameter values into the models. Examples are provided in comments within the dictionary.
6. Fill out the feature_transformation_args variable with any parameters you'd like to provide (namely the number of components ('n_components')
7. Run the program by typing "python main.py" in the console
