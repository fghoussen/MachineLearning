#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Read raw data.
    # https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/entrainez-un-modele-predictif-lineaire/TP_1_prostate_dataset.txt
    # https://rafalab.github.io/pages/649/prostate.html
    raw_data = pd.read_csv('prostate_dataset.txt', delimiter='\t')
    print('raw_data :\n', raw_data.head())

    # Extract data from dataset.
    x = raw_data[raw_data.columns[1:-3]].values # Dataset: variables.
    y = raw_data['lpsa'].values # Dataset: labels.
    print('x :\n', x[:5])
    print('y :\n', y[:5])

    # Split data set into training set and testing set.
    # https://openclassrooms.com/fr/courses/4011851-initiez-vous-au-machine-learning/4020631-exploitez-votre-jeu-de-donnees
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Change the hyperparameter alpha of the model to find the best one, compare different models (with/without regularization).
    # https://openclassrooms.com/fr/courses/4011851-initiez-vous-au-machine-learning/4022441-entrainez-votre-premier-k-nn
    # https://openclassrooms.com/fr/courses/4444646-entrainez-un-modele-predictif-lineaire/4507806-reduisez-le-nombre-de-variables-utilisees-par-votre-modele
    n_alphas = 100
    alphas = np.logspace(-5, 5, n_alphas) # alphas between 10^-5 and 10^5.
    models = []
    models.append((linear_model.LinearRegression(), 'linear reg')) # Baseline to compare to.
    models.append((linear_model.Ridge(), 'ridge')) # Compared to LinearRegression: Ridge reduces weights.
    models.append((linear_model.Lasso(fit_intercept=False), 'lasso')) # Compared to LinearRegression: Lasso can cancel some weights.
    models.append((linear_model.ElasticNet(), 'elastic net')) # Mixing Ridge (alpha) and Lasso (1. - alpha).
    error_min, best_alpha, best_model = float('inf'), 0, ''
    _, all_axis = plt.subplots(2, 4)
    for model_lbl in models:
        model, lbl = model_lbl[0], model_lbl[1]

        # Change the alpha hyperparameter.
        coefs, errors = [], []
        for a in alphas:
            if 'alpha' in model.get_params(): # LinearRegression has no alpha.
                model.set_params(alpha=a)

            # Scale data to reduce weights.
            # https://openclassrooms.com/fr/courses/4444646-entrainez-un-modele-predictif-lineaire/4507801-reduisez-l-amplitude-des-poids-affectes-a-vos-variables
            # https://openclassrooms.com/fr/courses/4297211-evaluez-les-performances-dun-modele-de-machine-learning/4308246-tp-selectionnez-le-nombre-de-voisins-dans-un-knn
            pipe = Pipeline([('scale', preprocessing.StandardScaler()), ('model', model)]) # Data scaling applied before / after any operator applied to the model.
            treg = TransformedTargetRegressor(regressor=pipe, transformer=preprocessing.MinMaxScaler()) # Target scaling applied before / after any operator applied to the model.

            # Train a model.
            treg.fit(x_train, y_train)
            coefs.append(treg.regressor_['model'].coef_) # LinearRegression will have always the same coefs.
            errors.append(np.mean((treg.predict(x_test) - y_test) ** 2)) # LinearRegression will have always the same error.
        # Plot errors.
        axis = all_axis.ravel()[0]
        axis.plot(alphas, errors, label=lbl)
        axis.set_xscale('log')
        axis.set_xlabel('alpha')
        axis.set_ylabel('errors')
        axis.legend()
        # Save best model / alpha.
        if np.min(errors) < error_min:
            error_min = np.min(errors)
            best_alpha = alphas[np.argmin(errors)]
            best_model = lbl
        # Plot weights.
        nb_coefs = np.shape(coefs)[1]
        for c in range(nb_coefs):
            axis = all_axis.ravel()[c+1]
            coef = np.array(coefs)[:, c]
            axis.plot(alphas, coef, label=lbl)
            axis.set_xscale('log')
            axis.set_xlabel('alpha')
            axis.set_ylabel('weights_'+str(c)+': '+raw_data.columns[1+c])
            axis.legend()
    for i in range(8):
        axis = all_axis.ravel()[i]
        axis.axvline(best_alpha, label='best: '+best_model, color='k', ls='--')
        axis.legend()
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
    plt.show()

if __name__ == '__main__':
    # https://openclassrooms.com/fr/courses/4444646-entrainez-un-modele-predictif-lineaire/4507811-tp-comparez-le-comportement-du-lasso-et-de-la-regression-ridge
    main()
