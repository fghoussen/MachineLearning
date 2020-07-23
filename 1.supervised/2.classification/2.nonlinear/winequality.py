#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import roc_curve, auc
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
    y = np.where(y<6, 0, 1) # Transform into binary classification problem: we want to identify "good" wine (quality > 6)
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
    models.append((svm.SVC(kernel='rbf'), 'SVM rbf')) # We use a gaussian kernel: 'rbf' radial basis function.
    models.append((svm.SVC(kernel='sigmoid'), 'SVM sigmoid')) # We use a sigmoid kernel.
    for idx_model, model_lbl in enumerate(models):
        # Train a model.
        model, lbl = model_lbl[0], model_lbl[1]
        for g in np.logspace(-2, 2, 3): # g coefficient between 10^-2 and 10^2.
            # Set parameter model.
            model.set_params(gamma=g)
            for c in np.logspace(-2, 2, 3): # c coefficient between 10^-2 and 10^2.
                # Set parameter model.
                model.set_params(C=c)
                # Feed the model.
                model.fit(x_train,y_train)
                # Get prediction for positive value
                y_prob = model.predict(x_test)
                # Compute ROC curve.
                false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
                roc_auc = auc(false_positive_rate, true_positive_rate)
                # Plot ROC to identify the best binary classifier.
                axis = plt.subplot(1, 2, idx_model+1)
                axis.set_title('Receiver Operating Characteristic')
                axis.plot(false_positive_rate,true_positive_rate, label='%s - C %08.3f - gamma %s - AUC = %0.5f'%(lbl, c, g, roc_auc))
                axis.set_ylabel('True Positive Rate')
                axis.set_xlabel('False Positive Rate')
        # Plot random binary classifier.
        axis = plt.subplot(1, 2, idx_model+1)
        axis.plot([0, 1], [0, 1], linestyle='--', label='random binary classifier', color='k')
        axis.legend(loc = 'lower right')
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
    plt.show()

if __name__ == '__main__':
    # https://openclassrooms.com/fr/courses/4470406-utilisez-des-modeles-supervises-non-lineaires/4722466-classifiez-vos-donnees-avec-une-svm-a-noyau
    main()
