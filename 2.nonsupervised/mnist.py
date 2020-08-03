#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Idea: first apply clustering to create groups, second apply reducing-dimension algorithm in order to better visualize data.

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, cluster, decomposition, manifold
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Get dataset.
    print('Fetching dataset...')
    mnist = fetch_openml('mnist_784', version=1)

    # Sampling the dataset: reduce the (too big) dataset, take one data every 50.
    data = mnist.data[::50, :]
    target = mnist.target[::50]
    nb_clusters = 10 # 10 clusters: mnist is a dataset of pictures from 0 to 9.

    # Split training and testing sets.
    x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.8)

    # Scale data to reduce weights.
    # https://openclassrooms.com/fr/courses/4444646-entrainez-un-modele-predictif-lineaire/4507801-reduisez-l-amplitude-des-poids-affectes-a-vos-variables
    std_scale = preprocessing.StandardScaler().fit(x_train)
    x_scaled = std_scale.transform(x_train)

    # First apply clustering to create groups.
    cls = cluster.KMeans(n_clusters=nb_clusters)
    cls.fit(x_scaled)

    # Second apply reducing-dimension algorithm in order to better visualize data.
    red_dim = []
    red_dim.append((decomposition.PCA(n_components=2), 'PCA')) # PCA.
    red_dim.append((manifold.Isomap(n_components=2), 'Isomap')) # globally preserving manifold.
    red_dim.append((manifold.TSNE(n_components=2, init='pca'), 't-SNE')) # locally preserving manifold.
    cmap = plt.cm.rainbow(np.linspace(0, 1, nb_clusters))
    for rd_lbl in red_dim:
        rd, lbl = rd_lbl
        print('performing ' + lbl + '...')
        x_projected = rd.fit_transform(x_scaled) # Project data on principal components.
        for idx, row in enumerate(x_projected):
            plt.text(row[0], row[1], y_train[idx], c=cmap[cls.labels_[idx]], wrap=True)
        plt.xlim(np.min(x_projected), np.max(x_projected))
        plt.ylim(np.min(x_projected), np.max(x_projected))
        plt.title('projected data with ' + lbl + ' (colored with cluster labels).')
        plt.show()

if __name__ == '__main__':
    # https://openclassrooms.com/fr/courses/4379436-explorez-vos-donnees-avec-des-algorithmes-non-supervises/6737881-entrainez-vous-a-manipuler-des-algorithmes-de-clustering-avec-sklearn
    main()
