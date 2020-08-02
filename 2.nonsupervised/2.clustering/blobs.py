#!/usr/bin/python
# -*- coding: UTF-8 -*-

from sklearn import datasets, preprocessing, cluster
import matplotlib.pyplot as plt

def main():
    # Create random data.
    n = 1500 # nb circles.
    for i, x_y in enumerate([datasets.make_circles(n, factor=.5, noise=.05), datasets.make_moons(n_samples=n, noise=.05)]):
        x, y = x_y

        # Scale data to reduce weights.
        # https://openclassrooms.com/fr/courses/4444646-entrainez-un-modele-predictif-lineaire/4507801-reduisez-l-amplitude-des-poids-affectes-a-vos-variables
        std_scale = preprocessing.StandardScaler().fit(x)
        x_scaled = std_scale.transform(x)

        # Perform DBSCAN on scaled data.
        range_eps = [0.05, 0.1, 0.2, 0.3]
        range_n_min = [5, 10, 20, 30]
        nb_plots = len(range_eps)+1 # +1: add true clusters.
        for j, eps_n_min in enumerate(zip(range_eps, range_n_min)):
            # Perform DBSCAN on scaled data.
            e, n_min = eps_n_min
            cls = cluster.DBSCAN(eps=e, min_samples=n_min)
            cls.fit(x_scaled)

            # Plot DBSCAN.
            axis = plt.subplot(2, nb_plots, 1+j+nb_plots*i)
            axis.scatter(x_scaled[:, 0], x_scaled[:, 1], c=cls.labels_, s=50)
            axis.set_title('eps %04.2f, n_min %02d' % (e, n_min))

        # Plot true clusters.
        axis = plt.subplot(2, nb_plots, nb_plots+nb_plots*i)
        axis.scatter(x_scaled[:, 0], x_scaled[:, 1], c=y, s=50)
        axis.set_title('true clusters')
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
    plt.suptitle('DBSCAN')
    plt.show()

if __name__ == '__main__':
    # https://openclassrooms.com/fr/courses/4379436-explorez-vos-donnees-avec-des-algorithmes-non-supervises/4379571-partitionnez-vos-donnees-avec-dbscan
    main()
