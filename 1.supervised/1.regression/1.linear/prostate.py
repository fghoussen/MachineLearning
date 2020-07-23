#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Read raw data.
    # https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/entrainez-un-modele-predictif-lineaire/TP_1_prostate_dataset.txt
    # https://rafalab.github.io/pages/649/prostate.html
    raw_data = pd.read_csv('prostate_dataset.txt', delimiter='\t')
    print('raw_data :\n', raw_data.head())

    # Extract data from dataset.
    x = raw_data.iloc[:, 1:-3] # Dataset: variables.
    y = raw_data.iloc[:, -2] # Dataset: labels.
    print('x :\n', x.head())
    print('y :\n', y.head())

    # Scale data to reduce weights.
    # https://openclassrooms.com/fr/courses/4444646-entrainez-un-modele-predictif-lineaire/4507801-reduisez-l-amplitude-des-poids-affectes-a-vos-variables
    std_scale = preprocessing.StandardScaler().fit(x)
    x_scaled = std_scale.transform(x)

    # Split data set into training set and testing set.
    # https://openclassrooms.com/fr/courses/4011851-initiez-vous-au-machine-learning/4020631-exploitez-votre-jeu-de-donnees
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2)

    # Change the hyperparameter alpha of the model to find the best one, compare different models (with/without regularization).
    # https://openclassrooms.com/fr/courses/4011851-initiez-vous-au-machine-learning/4022441-entrainez-votre-premier-k-nn
    # https://openclassrooms.com/fr/courses/4444646-entrainez-un-modele-predictif-lineaire/4507806-reduisez-le-nombre-de-variables-utilisees-par-votre-modele
    n_alphas = 100
    alphas = np.logspace(-5, 5, n_alphas) # alphas between 10^-5 and 10^5.
    models = []
    models.append((linear_model.LinearRegression(), 'linear reg'))
    models.append((linear_model.Ridge(), 'ridge')) # Compared to LinearRegression: Ridge reduces weights.
    models.append((linear_model.Lasso(fit_intercept=False), 'lasso')) # Compared to LinearRegression: Lasso can cancel some weights.
    models.append((linear_model.ElasticNet(), 'elastic net')) # Mixing Ridge (alpha) and Lasso (1. - alpha).
    error_min, best_alpha, best_model = float('inf'), 0, ''
    for model_lbl in models:
        # Train a model.
        model, lbl = model_lbl[0], model_lbl[1]
        coefs, errors = [], []
        for a in alphas:
            if 'alpha' in model.get_params(): # LinearRegression has no alpha.
                model.set_params(alpha=a)
            model.fit(x_train, y_train)
            coefs.append(model.coef_) # LinearRegression will have always the same coefs.
            errors.append(np.mean((model.predict(x_test) - y_test) ** 2)) # LinearRegression will have always the same error.
        # Plot errors.
        axis = plt.subplot(2, 4, 1)
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
            axis = plt.subplot(2, 4, c+2)
            coef = np.array(coefs)[:, c]
            axis.plot(alphas, coef, label=lbl)
            axis.set_xscale('log')
            axis.set_xlabel('alpha')
            axis.set_ylabel('weights_'+str(c)+': '+x.columns[c])
            axis.set_ylim([-0.8, 0.8]) # Set same ylim to see scale between weights.
            axis.legend()
    for i in range(8):
        axis = plt.subplot(2, 4, i+1)
        axis.axvline(best_alpha, label='best: '+best_model, color='k', ls='--')
        axis.legend()
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
    plt.show()

if __name__ == '__main__':
    # https://openclassrooms.com/fr/courses/4444646-entrainez-un-modele-predictif-lineaire/4507811-tp-comparez-le-comportement-du-lasso-et-de-la-regression-ridge
    main()
