#!/usr/bin/python
# -*- coding: UTF-8 -*-

from sklearn.datasets import make_circles
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

# https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html
def plot_svc_decision_function(model, axis):
    # Create grid to evaluate model.
    xlim = axis.get_xlim()
    ylim = axis.get_ylim()
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # Plot decision boundary and margins.
    axis.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    # Plot support vectors.
    axis.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                 s=100, facecolors='none', edgecolors='k', linewidths=1, alpha=0.5)

def main():
    # Create random data.
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html
    x, y = make_circles(200, factor=.1, noise=.1)

    # Scale data to reduce weights.
    # https://openclassrooms.com/fr/courses/4444646-entrainez-un-modele-predictif-lineaire/4507801-reduisez-l-amplitude-des-poids-affectes-a-vos-variables
    std_scale = preprocessing.StandardScaler().fit(x)
    x_scaled = std_scale.transform(x)

    # Split data set into training set and testing set.
    # https://openclassrooms.com/fr/courses/4011851-initiez-vous-au-machine-learning/4020631-exploitez-votre-jeu-de-donnees
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.5)

    # Change the hyperparameters of the model to find the best one, compare different models (with/without regularization).
    models = []
    models.append((svm.SVC(kernel='rbf'), 'SVM rbf')) # We use a gaussian kernel: 'rbf' radial basis function.
    models.append((svm.SVC(kernel='sigmoid'), 'SVM sigmoid')) # We use a sigmoid kernel.
    for idx_model, model_lbl in enumerate(models):
        # Train a model.
        model, lbl = model_lbl[0], model_lbl[1]
        best_roc_auc = 0.
        axis = plt.subplot(1, 2, idx_model+1)
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
                # Plot margins which enable decision.
                if roc_auc > best_roc_auc:
                    best_roc_auc = roc_auc
                    axis.clear() # Reset axis.
                    axis.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=50, cmap='autumn')
                    plot_svc_decision_function(model, axis)
                    axis.set_title('%s - C %08.3f - gamma %s - AUC = %0.5f'%(lbl, c, g, roc_auc))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
    plt.show()

if __name__ == '__main__':
    # https://openclassrooms.com/fr/courses/4470406-utilisez-des-modeles-supervises-non-lineaires/4722466-classifiez-vos-donnees-avec-une-svm-a-noyau
    main()
