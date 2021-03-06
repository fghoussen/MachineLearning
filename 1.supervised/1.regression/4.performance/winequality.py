#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import kernel_ridge
from sklearn.metrics import r2_score
from sklearn import dummy
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Read raw data.
    # https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality
    raw_data = pd.read_csv('winequality-white.csv', sep=';')
    print('raw_data :\n', raw_data.head())

    # Extract data from dataset.
    x = raw_data[raw_data.columns[:-1]].values # Dataset: variables.
    y = raw_data['quality'].values # Dataset: labels.
    print('x :\n', x[:5])
    print('y :\n', y[:5])

    # Scale data to reduce weights.
    # https://openclassrooms.com/fr/courses/4444646-entrainez-un-modele-predictif-lineaire/4507801-reduisez-l-amplitude-des-poids-affectes-a-vos-variables
    # https://openclassrooms.com/fr/courses/4297211-evaluez-les-performances-dun-modele-de-machine-learning/4308246-tp-selectionnez-le-nombre-de-voisins-dans-un-knn
    std_scale = preprocessing.StandardScaler().fit(x)
    x_scaled = std_scale.transform(x)
    std_scale = preprocessing.MinMaxScaler().fit(y.reshape(-1, 1))
    y_scaled = std_scale.transform(y.reshape(-1, 1)).ravel()

    for var, lbl in zip([x, x_scaled], ['not scaled', 'scaled']):
        fig, all_axis = plt.subplots(3, 4)
        for feat_idx in range(var.shape[1]):
            # variable alone.
            axis = all_axis.ravel()[feat_idx]
            axis.hist(var[:, feat_idx], bins=50)
            axis.set_title(raw_data.columns[feat_idx]+' - '+lbl, fontsize=14)
            # variable superimposed with others.
            last_axis = all_axis.ravel()[11]
            last_axis.hist(var[:, feat_idx], bins=50)
            last_axis.set_title('whole dataset - '+lbl, fontsize=14)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
        plt.show() # Show variable magnitude before / after scaling.

    # Split data set into training set and testing set.
    # https://openclassrooms.com/fr/courses/4011851-initiez-vous-au-machine-learning/4020631-exploitez-votre-jeu-de-donnees
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.3)

    # Fix hyper-parameters to test.
    param_grid = {
                'gamma': np.logspace(-2, 2, 6), # gamma coefficient between 10^-2 and 10^2.
                'alpha': np.logspace(-2, 2, 6), # alpha coefficient between 10^-2 and 10^2.
            }

    # Choose a score to optimize: r2 (coefficient of determination: regression score).
    score = 'r2'

    # Kernel ridge regressor: use cross validation to find the best hyper-parameters.
    clf = GridSearchCV(
        kernel_ridge.KernelRidge(kernel='rbf'), # Kernel ridge regressor.
        param_grid,     # hyper-parameters to test.
        cv=5,           # number of folds to test in cross validation.
        scoring=score   # score to optimize.
    )

    # Optimize best regressor on training set.
    clf.fit(x_train, y_train)

    # Print hyper-parameters.
    print("\nBest hyper-parameters on the training set:")
    print(clf.best_params_)

    # Print performances.
    print("\nCross validation results:")
    for mean, std, params in zip(clf.cv_results_['mean_test_score'], clf.cv_results_['std_test_score'], clf.cv_results_['params']):
        print("{} = {:.3f} (+/-{:.03f}) for {}".format(score, mean, std*2, params))

    # Print scores.
    # https://openclassrooms.com/fr/courses/4297211-evaluez-les-performances-dun-modele-de-machine-learning/4308276-evaluez-un-algorithme-de-regression
    y_pred = clf.predict(x_train)
    print("\nBest regressor score on training set: {:.3f}".format(r2_score(y_train, y_pred)))
    y_pred = clf.predict(x_test)
    print("\nBest regressor score on testing set: {:.3f}".format(r2_score(y_test, y_pred)))

    # Compare with baseline dummy regressor.
    best_dclf, best_dclf_score = None, -float('inf')
    for s in ['mean', 'median', 'quantile']:
        dclf = dummy.DummyRegressor(strategy=s, quantile=0.25)
        dclf.fit(x_train, y_train)
        dclf_score = r2_score(y_test, dclf.predict(x_test))
        if dclf_score > best_dclf_score:
            best_dclf, best_dclf_score = dclf, dclf_score
    y_pred = best_dclf.predict(x_train)
    print("\nBest dummy regressor score on training set: {:.3f}".format(r2_score(y_train, y_pred)))
    y_pred = best_dclf.predict(x_test)
    print("\nBest dummy regressor score on testing set: {:.3f}".format(r2_score(y_test, y_pred)))

if __name__ == '__main__':
    # https://openclassrooms.com/fr/courses/4297211-evaluez-les-performances-dun-modele-de-machine-learning/4308246-tp-selectionnez-le-nombre-de-voisins-dans-un-knn
    main()
