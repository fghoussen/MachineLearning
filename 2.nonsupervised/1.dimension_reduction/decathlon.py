#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
from sklearn import preprocessing
from sklearn import decomposition
import matplotlib.pyplot as plt

def main():
    # Read raw data.
    # http://factominer.free.fr/factomethods/datasets/decathlon.txt
    raw_data = pd.read_csv('decathlon.txt', delimiter='\t')
    print('raw_data :\n', raw_data.head())

    # Drop data we don't need.
    data = raw_data.drop(['Points', 'Rank', 'Competition'], axis=1)

    # Extract data from dataset.
    x = data.values
    print('x :\n', x[:5])

    # Scale data to reduce weights.
    # https://openclassrooms.com/fr/courses/4444646-entrainez-un-modele-predictif-lineaire/4507801-reduisez-l-amplitude-des-poids-affectes-a-vos-variables
    std_scale = preprocessing.StandardScaler().fit(x)
    x_scaled = std_scale.transform(x)

    # Perform PCA on scaled data.
    pca = decomposition.PCA(n_components=2)
    pca.fit(x_scaled)
    print('percentage of explained variance: ', pca.explained_variance_ratio_.sum())

    # Project data on principal components.
    x_projected = pca.transform(x_scaled)

    # Plot contribution of each variable to each component.
    pcs = pca.components_
    for i, (x, y) in enumerate(zip(pcs[0, :], pcs[1, :])):
        plt.plot([0, x], [0, y], color='k')
        plt.text(x, y, data.columns[i], fontsize='14')
    plt.plot([-0.7, 0.7], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-0.7, 0.7], color='grey', ls='--')
    plt.title('contribution of each variable to each component')
    plt.show()

    # Plot projected data (colored with 'Rank' data).
    plt.scatter(x_projected[:, 0], x_projected[:, 1], c=raw_data.get('Rank'))
    plt.plot([-5, 5], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-5, 5], color='grey', ls='--')
    plt.title('projected data (colored with rank).')
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    # https://openclassrooms.com/fr/courses/4379436-explorez-vos-donnees-avec-des-algorithmes-non-supervises/4379506-tp-acp-d-un-jeu-de-donnees-sur-les-performances-d-athletes-olympiques
    main()
