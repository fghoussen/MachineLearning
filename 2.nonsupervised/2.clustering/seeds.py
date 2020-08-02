#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
from sklearn import preprocessing
from sklearn import cluster, metrics
from sklearn import decomposition
import matplotlib.pyplot as plt

def main():
    # Read raw data.
    # http://archive.ics.uci.edu/ml/datasets/seeds
    raw_data = pd.read_csv('seeds_dataset.txt', delim_whitespace=True, header=None)
    print('raw_data :\n', raw_data.head())

    # Drop data we don't need.
    data = raw_data.drop(7, axis=1)

    # Extract data from dataset.
    x = data.values
    y = raw_data[7].values # Real true clusters extracted from dataset.
    print('x :\n', x[:5])

    # Scale data to reduce weights.
    # https://openclassrooms.com/fr/courses/4444646-entrainez-un-modele-predictif-lineaire/4507801-reduisez-l-amplitude-des-poids-affectes-a-vos-variables
    std_scale = preprocessing.StandardScaler().fit(x)
    x_scaled = std_scale.transform(x)

    # Test k-means with different cluster numbers: check the most relevant (= biggest silhouette).
    silhouette = []
    range_clusters = range(2, 10)
    for num_clusters in range_clusters:
        cls = cluster.KMeans(n_clusters=num_clusters, n_init=1, init='random')
        cls.fit(x_scaled)
        sil = metrics.silhouette_score(x_scaled, cls.labels_)
        silhouette.append(sil)

    # Plot k-means silhouette.
    plt.plot(range_clusters, silhouette, marker='o')
    plt.xlabel('k')
    plt.ylabel('silhouette')
    plt.title('k-means clustering')
    plt.show()

    # Plot k-means with different cluster numbers.
    k_max = 5
    range_clusters = range(2, k_max)
    for idx, num_clusters in enumerate(range_clusters):
        cls = cluster.KMeans(n_clusters=num_clusters, n_init=1, init='random')
        cls.fit(x_scaled)

        # Perform PCA on scaled data.
        pca = decomposition.PCA(n_components=num_clusters)
        pca.fit(x_scaled)
        print('percentage of explained variance: ', pca.explained_variance_ratio_.sum())

        # Project data on principal components.
        x_projected = pca.transform(x_scaled)

        # Plot.
        axis = plt.subplot(len(range_clusters), 2, 2*idx+1)
        axis.scatter(x_projected[:, 0], x_projected[:, 1], c=cls.labels_)
        ari = metrics.adjusted_rand_score(y, cls.labels_)
        axis.set_title(str(num_clusters)+'-means clustering: score %.2f' % ari)
    axis = plt.subplot(len(range_clusters), 2, 4)
    axis.scatter(x_projected[:, 0], x_projected[:, 1], c=y)
    axis.set_title('raw data true clustering')
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
    plt.show()

if __name__ == '__main__':
    # https://openclassrooms.com/fr/courses/4379436-explorez-vos-donnees-avec-des-algorithmes-non-supervises/4379566-partitionnez-vos-donnees-avec-l-algorithme-du-k-means
    main()
