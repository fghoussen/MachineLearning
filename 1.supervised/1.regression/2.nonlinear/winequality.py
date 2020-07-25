#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import kernel_ridge
from sklearn.metrics import mean_squared_error
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
    std_scale = preprocessing.StandardScaler().fit(x)
    x_scaled = std_scale.transform(x)

    # Split data set into training set and testing set.
    # https://openclassrooms.com/fr/courses/4011851-initiez-vous-au-machine-learning/4020631-exploitez-votre-jeu-de-donnees
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3)

    # Change the hyperparameters of the model to find the best one, compare different models (with/without regularization).
    models = []
    models.append((kernel_ridge.KernelRidge(kernel='rbf'), 'reg ridge rbf')) # We use a gaussian kernel: 'rbf' radial basis function.
    for idx_model, model_lbl in enumerate(models):
        # Train a model.
        model, lbl = model_lbl[0], model_lbl[1]
        best_rmse, best_g, best_a = float('inf'), 0, 0
        worst_rmse, worst_g, worst_a = 0, 0, 0
        all_g, all_a, all_rmse = [], [], []
        for g in np.logspace(-2, 2, 6): # g coefficient between 10^-2 and 10^2.
            # Set parameter model.
            model.set_params(gamma=g)
            for a in np.logspace(-2, 2, 6): # a coefficient between 10^-2 and 10^2.
                # Set parameter model.
                model.set_params(alpha=a)
                # Feed the model.
                model.fit(x_train,y_train)
                # Get prediction for positive value
                y_prob = model.predict(x_test)
                # Compute root mean square error.
                rmse = np.sqrt(mean_squared_error(y_test, y_prob))
                all_g.append(g)
                all_a.append(a)
                all_rmse.append(rmse)
                # Save best and worst models.
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_g = g
                    best_a = a
                if rmse > worst_rmse:
                    worst_rmse = rmse
                    worst_g = g
                    worst_a = a
        # Plot random binary classifier.
        axis = plt.subplot(1, 2, idx_model+1, projection='3d')
        axis.set_xlabel('gamma')
        axis.set_ylabel('alpha')
        axis.set_zlabel('rms error')
        axis.scatter3D(all_g, all_a, all_rmse)
        # Get the best and worst model.
        for g, a in zip([best_g, worst_g], [best_a, worst_a]):
            model.set_params(gamma=g)
            model.set_params(alpha=a)
            model.fit(x_train,y_train)
            y_prob = model.predict(x_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_prob))
            # Plot true versus predicted score (marker size = number of pairs true/predicted = the bigger, the better).
            sizes = {}
            for (yt, yp) in zip(list(y_test), list(y_prob)):
                if (yt, yp) in sizes.keys():
                    sizes[(yt, yp)] += 1
                else:
                    sizes[(yt, yp)] = 1
            keys = sizes.keys()
            axis = plt.subplot(1, 2, idx_model+2)
            axis.scatter([k[0] for k in keys], [k[1] for k in keys],
                         s=[sizes[k] for k in keys], # marker size = number of pairs (true, predicted) = the bigger, the better.
                         label='alpha %08.3f - gamma %08.3f - RMSE = %0.5f'%(a, g, rmse))
            axis.set_xlabel('True score')
            axis.set_ylabel('Predicted score')
            axis.set_title('best kernel Ridge Regression')
            axis.legend()
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
    plt.show()

if __name__ == '__main__':
    # https://openclassrooms.com/fr/courses/4470406-utilisez-des-modeles-supervises-non-lineaires/4729316-apprenez-des-etiquettes-reelles-avec-une-regression-ridge-a-noyau
    main()
