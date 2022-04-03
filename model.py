from sklearn import tree
from sklearn.model_selection import learning_curve
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

class Model:

    def __init__(self, model_type='dtree', **kwargs):
        self.model_type = 'dtree'
        self.model = self.build_model(model_type, **kwargs)

    def build_model(self, model_type, **kwargs):

        model = None

        match model_type:

            case 'dtree':

                # create decision tree classifier
                # reference: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
                # -the default values for the sklearn decision tree use the gini index to perform best split
                # --are limited to 
                model = tree.DecisionTreeClassifier(**kwargs)
                self.model_type = 'Decision Tree'

            case 'nn':
                model = MLPClassifier(**kwargs)
                self.model_type = 'Neural Network'
                self.problem_type = 'supervized'

            case 'boosted-dtree':
                classifier = tree.DecisionTreeClassifier(**kwargs['classifier'])
                model = AdaBoostClassifier(base_estimator=classifier, **kwargs['booster'])
                self.model_type = 'Boosted Decision Tree'
                self.problem_type = 'supervized'

            case 'svm':
                model = SVC(**kwargs)
                self.model_type = 'SVM'
                self.problem_type = 'supervized'

            case 'knn':
                model = KNeighborsClassifier(**kwargs)
                self.model_type = 'KNN'
                self.problem_type = 'supervized'

            case 'kmeans':
                model = KMeans(**kwargs)
                self.model_type = 'K-Means'
                self.problem_type = 'unsupervized'

            case 'em':
                model = GaussianMixture(**kwargs)
                self.model_type = 'Expectation Maximization'
                self.problem_type = 'unsupervized'

        self.model = model

        return model

    def compile_stats(self, X, y, cv_folds) :

        #reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
        #reference: https://zahidhasan.github.io/2020/10/13/bias-variance-trade-off-and-learning-curve.html

        #initialize matplotlib figure
        #fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        #grab learning curve data
        train_size, train_scores, test_scores = learning_curve(self.model, X, y, cv=cv_folds)

        train_score_mean = np.mean(train_scores, axis=1)
        test_score_mean = np.mean(test_scores, axis=1)

        plt.plot(train_size, train_score_mean, color="b")
        plt.plot(train_size, test_score_mean, color="g")
        plt.legend(('Training Accuracy', 'CV Accuracy'))
        plt.xlabel('Training Samples')
        plt.ylabel('Accuracy Score')
        plt.title(f"Learning Curve for {self.model_type}")
        plt.show()
        plt.clf()
        plt.cla()

    def draw_tree(self):
        if self.model_type == 'Decision Tree':
            tree.plot_tree(self.model, max_depth=3)
            plt.show()

    def print_nn_stats(self):
        plt.plot(self.model.loss_curve_)
        plt.title('Neural Network Loss Curve')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.show()

    def get_class_accuracy(self, y_test, pred):
        
        class_accuracy = {}
        for y in zip(y_test, pred):
            if y[0] in class_accuracy:
                class_accuracy[y[0]]['correct'] += int(y[0] == y[1])
                class_accuracy[y[0]]['total'] += 1
            else:
                class_accuracy[y[0]] = {
                    "total": 1,
                    "correct": int(y[0] == y[1])
                }

        return class_accuracy
