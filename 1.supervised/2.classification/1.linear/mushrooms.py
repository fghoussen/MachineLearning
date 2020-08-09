#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model, svm
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Read raw data.
    # https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/entrainez-un-modele-predictif-lineaire/TP_2_datset_mushrooms.csv
    raw_data = pd.read_csv('mushrooms_dataset.csv')
    print('raw_data :\n', raw_data.head())

    # Convert letters to numbers : machine learning algorithms works only on numbers.
    labelencoder=preprocessing.LabelEncoder()
    for col in raw_data.columns:
        raw_data[col] = labelencoder.fit_transform(raw_data[col])
    print('converted raw_data :\n', raw_data.head())

    # Extract data from dataset.
    x = raw_data.iloc[:, 1:23] # Dataset: variables.
    y = raw_data.iloc[:, 0] # Dataset: labels.
    print('x :\n', x.head())
    print('y :\n', y.head())

    # Scale data to reduce weights.
    # https://openclassrooms.com/fr/courses/4444646-entrainez-un-modele-predictif-lineaire/4507801-reduisez-l-amplitude-des-poids-affectes-a-vos-variables
    std_scale = preprocessing.StandardScaler().fit(x)
    x_scaled = std_scale.transform(x)

    # Split data set into training set and testing set.
    # https://openclassrooms.com/fr/courses/4011851-initiez-vous-au-machine-learning/4020631-exploitez-votre-jeu-de-donnees
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3)

    # Change the hyperparameters of the model to find the best one, compare different models (with/without regularization).
    models = []
    models.append((linear_model.LogisticRegression(solver = 'liblinear'), 'logistic reg'))
    models.append((svm.LinearSVC(loss='hinge'), 'SVM'))
    _, all_axis = plt.subplots(1, 2)
    for idx_model, model_lbl in enumerate(models):
        # Train a model.
        model, lbl = model_lbl[0], model_lbl[1]
        axis = all_axis.ravel()[idx_model]
        for p in ['l1', 'l2']:
            # Set parameter model.
            model.set_params(penalty=p)
            if p == 'l1' and isinstance(model, svm.LinearSVC):
                continue # Not supported.
            for c in np.logspace(-3, 3, 7): # c coefficient between 10^-3 and 10^3.
                # Set parameter model.
                model.set_params(C=c)
                # Feed the model.
                model.fit(x_train,y_train)
                # Get prediction for positive value
                y_prob = None
                if isinstance(model, linear_model.LogisticRegression):
                    y_prob = model.predict_proba(x_test)[:,1]
                if isinstance(model, svm.LinearSVC):
                    y_prob = model.predict(x_test)
                # Compute ROC curve.
                false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
                roc_auc = auc(false_positive_rate, true_positive_rate)
                # Plot ROC to identify the best binary classifier.
                axis.set_title('Receiver Operating Characteristic')
                axis.plot(false_positive_rate,true_positive_rate, label='%s - C %08.3f - penalty %s - AUC = %0.5f'%(lbl, c, p, roc_auc))
                axis.set_ylabel('True Positive Rate')
                axis.set_xlabel('False Positive Rate')
        # Plot random binary classifier.
        axis.plot([0, 1], [0, 1], linestyle='--', label='random binary classifier', color='k')
        axis.legend(loc = 'lower right')
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
    plt.show()

if __name__ == '__main__':
    # https://openclassrooms.com/fr/courses/4444646-entrainez-un-modele-predictif-lineaire/4507851-tp-entrainez-une-regression-logistique-et-une-svm-lineaire
    main()
