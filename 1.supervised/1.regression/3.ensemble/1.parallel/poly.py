#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
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

    # Create bagging and random forest models.
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    regressors = [BaggingRegressor(n_estimators=5, base_estimator=KNeighborsRegressor()),
                  BaggingRegressor(n_estimators=5, base_estimator=SVR()),
                  BaggingRegressor(n_estimators=5, base_estimator=KernelRidge()),
                  RandomForestRegressor(n_estimators=5)]
    for ax, reg in zip(axes.ravel(), regressors):
        # Set title.
        title = reg.__class__.__name__
        reg_params = reg.get_params()
        if 'base_estimator' in reg_params: # RandomForestRegressor has no 'base_estimator'.
            title += ', estimator: '+reg_params['base_estimator'].__class__.__name__
        ax.set_title(title)

        # Plot random data.
        ax.plot(x, y, 'o', color='black', markersize=2, label='random data')

        # Create augmented data : add dimensions to initial data in order to fit y as a polynomial of degree 5.
        x_augmented = np.array([x, x**2, x**3, x**4, x**5]).T

        # Train model.
        reg.fit(x_augmented, y)

        # Plot intermediate regression estimations.
        for i, tree in enumerate(reg.estimators_):
            ax.plot(x_augmented[:, 0], tree.predict(x_augmented), '-', color='gray', label='tree '+str(i))
            ax.axis('off')
            ax.legend()

        # Plot final regression.
        ax.plot(x_augmented[:, 0], reg.predict(x_augmented), '-', color='red', label=reg.__class__.__name__)
        ax.axis('off')
        ax.legend()
    plt.show()

if __name__ == '__main__':
    # https://openclassrooms.com/fr/courses/4470521-modelisez-vos-donnees-avec-les-methodes-ensemblistes/4664687-controlez-la-variance-a-l-aide-du-bagging
    main()
