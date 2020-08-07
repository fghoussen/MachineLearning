#!/usr/bin/python
# -*- coding: UTF-8 -*-

from sklearn import datasets, preprocessing, cluster
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, colorConverter
cm2 = ListedColormap(['#0000aa', '#ff2020'])

def plot_tree_partition(tree, X, y_true, ax):
    # Plot predicted contours.
    eps = X.std() / 2.
    x_min, x_max = X[:, 0].min() - eps, X[:, 0].max() + eps
    y_min, y_max = X[:, 1].min() - eps, X[:, 1].max() + eps
    xx = np.linspace(x_min, x_max, 1000)
    yy = np.linspace(y_min, y_max, 1000)
    X1, X2 = np.meshgrid(xx, yy)
    X_grid = np.c_[X1.ravel(), X2.ravel()]
    y_predict = tree.predict(X_grid)
    y_predict = y_predict.reshape(X1.shape)
    faces = tree.apply(X_grid)
    faces = faces.reshape(X1.shape)
    ax.contour(X1, X2, y_predict, colors='black', levels=[.5])
    ax.contourf(X1, X2, y_predict, alpha=.3, cmap=cm2, levels=[0, .5, 1])

    # Plot true points.
    ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap=cm2, s=50)

    # Set limits, remove ticks.
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())

def plot_classifier_partition(classifier, X, y_true, ax, threshold=0.5):
    # Plot predicted contours.
    eps = X.std() / 2.
    x_min, x_max = X[:, 0].min() - eps, X[:, 0].max() + eps
    y_min, y_max = X[:, 1].min() - eps, X[:, 1].max() + eps
    xx = np.linspace(x_min, x_max, 1000)
    yy = np.linspace(y_min, y_max, 1000)
    X1, X2 = np.meshgrid(xx, yy)
    X_grid = np.c_[X1.ravel(), X2.ravel()]
    try:
        decision_values = classifier.decision_function(X_grid)
        levels = [0] if threshold is None else [threshold]
        fill_levels = [decision_values.min()] + levels + [decision_values.max()]
    except AttributeError:
        # no decision_function
        decision_values = classifier.predict_proba(X_grid)[:, 1]
        levels = [.5] if threshold is None else [threshold]
        fill_levels = [0] + levels + [1]
    ax.contour(X1, X2, decision_values.reshape(X1.shape), colors='black', levels=levels)
    ax.contourf(X1, X2, decision_values.reshape(X1.shape), levels=fill_levels, alpha=.3, cmap=cm2)

    # Plot true points.
    ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap=cm2, s=50)

    # Set limits, remove ticks.
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())

def main():
    # Create random data.
    x, y = datasets.make_moons(n_samples=100, noise=0.25)

    # Scale data to reduce weights.
    # https://openclassrooms.com/fr/courses/4444646-entrainez-un-modele-predictif-lineaire/4507801-reduisez-l-amplitude-des-poids-affectes-a-vos-variables
    std_scale = preprocessing.StandardScaler().fit(x)
    x_scaled = std_scale.transform(x)

    # Split data set into training set and testing set.
    # https://openclassrooms.com/fr/courses/4011851-initiez-vous-au-machine-learning/4020631-exploitez-votre-jeu-de-donnees
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, stratify=y)

    # Create bagging and random forest models.
    classifiers = [BaggingClassifier(n_estimators=5), RandomForestClassifier(n_estimators=5)]
    for cls in classifiers:
        cls.fit(x_train, y_train)

        # Plot intermediate trees.
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        for i, (ax, tree) in enumerate(zip(axes.ravel(), cls.estimators_)):
            ax.set_title("Tree {}".format(i))
            plot_tree_partition(tree, x_train, y_train, ax)

        # Plot final classification.
        cls_axis = axes[-1, -1] # axis located in last row / column.
        cls_axis.set_title(cls.__class__.__name__)
        plot_classifier_partition(cls, x_train, y_train, cls_axis, threshold=0.5)
        fig.suptitle(cls.__class__.__name__)
    plt.show()

if __name__ == '__main__':
    # https://openclassrooms.com/fr/courses/4470521-modelisez-vos-donnees-avec-les-methodes-ensemblistes/4664687-controlez-la-variance-a-l-aide-du-bagging
    main()
