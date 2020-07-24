#!/usr/bin/python
# -*- coding: UTF-8 -*-

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Get dataset.
    print('Fetching dataset...')
    mnist = fetch_openml('mnist_784', version=1)

    # Sampling the dataset: reduce the (too big) dataset (k-NN is slow).
    mnist_size = mnist.data.shape[0] # Big data set: reduce it's size.
    sample = np.random.randint(mnist_size, size=mnist_size//10)
    data = mnist.data[sample]
    target = mnist.target[sample]

    # Split training and testing sets.
    x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.8)

    # Change hyperparameter k to get the best k-NN.
    errors, best_error, best_k = [], float('inf'), 0
    for k in range(2,15):
        # Feed k-NN model: learn from training set.
        print('Evaluating %s-NN...'%k)
        knn = neighbors.KNeighborsClassifier(k)
        knn.fit(x_train, y_train)

        # Check errors on testing set.
        error = 100*(1 - knn.score(x_test, y_test))
        errors.append(error)
        if error < best_error:
            best_error = error
            best_k = k
    plt.plot(range(2,15), errors, 'o-')
    plt.xlabel('k')
    plt.ylabel('error')
    plt.title('k-NN error')
    plt.show()

    # Get the best k-NN.
    print('Using best classifier: %s-NN...'%best_k)
    knn = neighbors.KNeighborsClassifier(best_k)
    knn.fit(x_train, y_train)

    # Get k-NN predictions.
    predicted = knn.predict(x_test)
    good = (y_test == predicted) # Good prediction.
    wrong = (y_test != predicted) # Wrong prediction.
    good_predicted = predicted[good] # Prediction OK.
    wrong_predicted = predicted[wrong] # Prediction KO.

    # Resize data as images.
    images = x_test.reshape((-1, 28, 28))
    good_images = images[good,:,:]
    wrong_images = images[wrong,:,:]

    for images, predicted, lbl in zip([good_images, wrong_images], [good_predicted, wrong_predicted], ['good', 'wrong']):
        # Select randomly 12 images.
        select = np.random.randint(images.shape[0], size=12)

        # Show images with associated prediction.
        plt.subplots(3,4)
        for index, value in enumerate(select):
            plt.subplot(3, 4, index+1)
            plt.axis('off')
            plt.imshow(images[value], cmap=plt.cm.gray_r, interpolation="nearest")
            plt.title('Predicted: {}'.format(predicted[value]))
        plt.suptitle(lbl+' predictions')
        plt.show()

if __name__ == '__main__':
    # https://openclassrooms.com/fr/courses/4011851-initiez-vous-au-machine-learning/4022441-entrainez-votre-premier-k-nn
    main()
