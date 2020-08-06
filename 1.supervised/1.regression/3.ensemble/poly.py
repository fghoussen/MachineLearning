#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

def f(x):
    return x*np.cos(x) + np.random.normal(size=500)*2

def main():
    # Generate random data : create random points, and, keep only a subset of them.
    x = np.linspace(0, 10, 500)
    rng = np.random.RandomState(0)
    rng.shuffle(x)
    x = np.sort(x[:])
    y = f(x)

    # Run bagging with different estimators.
    base_estimators = [DecisionTreeRegressor(), KNeighborsRegressor(), SVR(), KernelRidge()]
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    for ax, base_estimator in zip(axes.ravel(), base_estimators):
        ax.set_title('estimator: '+str(base_estimator.__class__.__name__))

        # Plot random data.
        ax.plot(x, y, 'o', color='black', markersize=5, label='random data')

        # Create augmented data : add dimensions to initial data in order to fit y as a polynomial of degree 5.
        x_augmented = np.array([x, x**2, x**3, x**4, x**5]).T

        # Create bagging model.
        bagging_reg = BaggingRegressor(base_estimator=base_estimator, n_estimators=5)
        bagging_reg.fit(x_augmented, y)

        # Plot bagging trees.
        for i, tree in enumerate(bagging_reg.estimators_):
            ax.plot(x_augmented[:, 0], tree.predict(x_augmented), '-', color='gray', label='tree '+str(i))
            ax.axis('off')
            ax.legend()

        # Plot bagging classification.
        bagging_axis = axes[-1, -1] # axis located in last row / column.
        ax.plot(x_augmented[:, 0], bagging_reg.predict(x_augmented), '-', color='red', label='bagging')
        ax.axis('off')
        ax.legend()
    plt.show()

if __name__ == '__main__':
    # https://openclassrooms.com/fr/courses/4470521-modelisez-vos-donnees-avec-les-methodes-ensemblistes/4664687-controlez-la-variance-a-l-aide-du-bagging
    main()
