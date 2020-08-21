#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, neighbors

def f(x):
    return x*np.cos(x) + np.random.normal(size=500)*2

def main():
    # Generate random data : create random points, and, keep only a subset of them.
    x = np.linspace(0, 10, 500)
    rng = np.random.RandomState(0)
    rng.shuffle(x)
    x = np.sort(x[:])
    y = f(x)

    # Plot random data.
    plt.plot(x, y, 'o', color='black', markersize=2, label='random data')

    # Create augmented data : add dimensions to initial data in order to fit y as a polynomial of degree 5.
    x_augmented = np.array([x, x**2, x**3, x**4, x**5]).T

    # Polynomial regression : regression on augmented data.
    regrs = []
    regrs.append((linear_model.LinearRegression(), 'polynomial reg'))
    regrs.append((neighbors.KNeighborsRegressor(15), '15-NN reg'))
    for regr in regrs:
        model, lbl = regr[0], regr[1]

        # Train model.
        model.fit(x_augmented, y)
        # Plot regression.
        plt.plot(x_augmented[:,0], model.predict(x_augmented), '-', label=lbl)
    plt.axis('off')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # https://openclassrooms.com/fr/courses/4379436-explorez-vos-donnees-avec-des-algorithmes-non-supervises/4379516-decouvrez-la-reduction-dimensionnelle-non-lineaire
    main()
