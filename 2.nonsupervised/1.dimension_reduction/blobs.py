#!/usr/bin/python
# -*- coding: UTF-8 -*-

from sklearn.datasets.samples_generator import make_circles
from sklearn import preprocessing
from sklearn import decomposition
import matplotlib.pyplot as plt

def main():
    # Create random data.
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html
    n = 200 # nb circles.
    x, y = make_circles(n, factor=.1, noise=.1)

    # Scale data to reduce weights.
    # https://openclassrooms.com/fr/courses/4444646-entrainez-un-modele-predictif-lineaire/4507801-reduisez-l-amplitude-des-poids-affectes-a-vos-variables
    std_scale = preprocessing.StandardScaler().fit(x)
    x_scaled = std_scale.transform(x)

    for i, g in enumerate([1, 10, 100]):
        # Perform kernel PCA on scaled data.
        kpca = decomposition.KernelPCA(n_components=1, kernel='rbf', gamma=g)
        kpca.fit(x_scaled)

        # Project data on principal components.
        x_kpca = kpca.transform(x_scaled)

        # Plot.
        axis = plt.subplot(3, 2, 1+2*i)
        axis.scatter(x_scaled[:, 0], x_scaled[:, 1], c=x_kpca, s=50)
        axis.set_title('initial space, g %03d'%g)
        axis = plt.subplot(3, 2, 2+2*i)
        axis.scatter(x_kpca, [0]*n, c=x_kpca, s=50)
        axis.set_title('redescription space, g %03d'%g)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
    plt.suptitle('kPCA rbf')
    plt.show()

if __name__ == '__main__':
    # https://openclassrooms.com/fr/courses/4470406-utilisez-des-modeles-supervises-non-lineaires/4722466-classifiez-vos-donnees-avec-une-svm-a-noyau
    main()
