#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

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
    models = [BaggingRegressor(n_estimators=5, base_estimator=KNeighborsRegressor()),
              BaggingRegressor(n_estimators=5, base_estimator=SVR()),
              BaggingRegressor(n_estimators=5, base_estimator=KernelRidge(kernel='rbf')),
              RandomForestRegressor(n_estimators=5)]
    for axis, model in zip(axes.ravel(), models):
        # Set title.
        title = model.__class__.__name__
        mdl_params = model.get_params()
        if 'base_estimator' in mdl_params: # RandomForestRegressor has no 'base_estimator'.
            title += ', estimator: '+mdl_params['base_estimator'].__class__.__name__
        axis.set_title(title)

        # Plot random data.
        axis.plot(x, y, 'o', color='black', markersize=2, label='random data')

        # Create augmented data : add dimensions to initial data in order to fit y as a polynomial of degree 5.
        x_augmented = np.array([x, x**2, x**3, x**4, x**5]).T

        # Scale data to reduce weights.
        # https://openclassrooms.com/fr/courses/4444646-entrainez-un-modele-predictif-lineaire/4507801-reduisez-l-amplitude-des-poids-affectes-a-vos-variables
        # https://openclassrooms.com/fr/courses/4297211-evaluez-les-performances-dun-modele-de-machine-learning/4308246-tp-selectionnez-le-nombre-de-voisins-dans-un-knn
        pipe = Pipeline([('scale', preprocessing.StandardScaler()), ('model', model)]) # Data scaling applied before / after any operator applied to the model.
        y_transformer = preprocessing.MinMaxScaler().fit(y.reshape(-1, 1))
        treg = TransformedTargetRegressor(regressor=pipe, transformer=y_transformer) # Target scaling applied before / after any operator applied to the model.

        # Train model.
        treg.fit(x_augmented, y)

        # Plot intermediate regression estimations.
        for i, tree in enumerate(treg.regressor_['model'].estimators_):
            x_augmented_scaled = treg.regressor_['scale'].transform(x_augmented) # x input after scaling (as tree does not use Pipeline).
            y_hat = tree.predict(x_augmented_scaled) # y outcome before scaling (as tree does not use TransformedTargetRegressor).
            y_pred = y_transformer.inverse_transform(y_hat.reshape(-1, 1))

            axis.plot(x, y_pred, '--', label='tree '+str(i))
            axis.axis('off')
            axis.legend()

        # Plot final regression.
        axis.plot(x, treg.predict(x_augmented), '-', color='black', label=model.__class__.__name__)
        axis.axis('off')
        axis.legend()
    plt.show()

if __name__ == '__main__':
    # https://openclassrooms.com/fr/courses/4470521-modelisez-vos-donnees-avec-les-methodes-ensemblistes/4664687-controlez-la-variance-a-l-aide-du-bagging
    main()
