#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

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

    # Polynomial regression : linear regression on augmented data.
    regr = linear_model.LinearRegression()
    regr.fit(x_augmented, y)

    # Plot polynomial regression.
    plt.plot(x_augmented[:,0], regr.predict(x_augmented), '-', color='orange', label='polynomial regression')
    plt.axis('off')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # https://openclassrooms.com/fr/courses/4379436-explorez-vos-donnees-avec-des-algorithmes-non-supervises/4379516-decouvrez-la-reduction-dimensionnelle-non-lineaire
    main()
