# Classify Flowers Using Measurements
# The test problem we will be using in this tutorial is iris classification.
#
# The problem is comprised of 150 observations of iris flowers from three different species. There are 4 measurements of given flowers: sepal length, sepal width, petal length and petal width, all in the same unit of centimeters. The predicted attribute is the species, which is one of setosa, versicolor or virginica.
#
# It is a standard dataset where the species is known for all instances. As such we can split the data into training and test datasets and use the results to evaluate our algorithm implementation. Good classification accuracy on this problem is above 90% correct, typically 96% or better.
#
# You can download the dataset for free from iris.data, see the resources section for further details.
#
# How to implement k-Nearest Neighbors in Python
# This tutorial is broken down into the following steps:
#
# 1. Handle Data: Open the dataset from CSV and split into test/train datasets.
# 2. Similarity: Calculate the distance between two data instances.
# 3. Neighbors: Locate k most similar data instances.
# 4. Response: Generate a response from a set of data instances.
# 5. Accuracy: Summarize the accuracy of predictions.
# 6. Main: Tie it all together.

import csv
import random
import math
import operator


def load_dataset(filename, split, trainingSet=[], testSet=[]):
    with open(file=filename, mode='r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


def euclidean_distance(instance1, instance2, length):
    '''
    We use euclidean distance here for calculating the distance between instances.
    :param instance1:
    :param instance2:
    :param length:
    :return:
    '''
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def get_neighbors(training_set, test_instance, k):
    distances = []
    length = len(test_instance) - 1
    for x in range(len(training_set)):
        dist = euclidean_distance(test_instance, training_set[x], length)
        distances.append((training_set[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def get_responses(neighbors):
    class_votes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1

    sortedValues = sorted(class_votes, key=operator.itemgetter(1), reverse=True)
    return sortedValues


def get_accuracy(test_set, predictions):
    correct = 0
    for x in range(len(test_set)):
        # print(test_set[x][-1], predictions[x])
        if test_set[x][-1] in predictions[x]:
            correct += 1
    # print(correct)
    return (correct / float(len(test_set))) * 100.0


def main():
    # prepare data
    training_set = []
    test_set = []
    split = 0.67
    load_dataset('../data/iris.data', split, training_set, test_set)
    # print(training_set)
    # print(test_set)
    # generate predictions
    predictions = []
    k = 3
    for x in range(len(test_set)):
        neighbors = get_neighbors(training_set, test_set[x], k)
        result = get_responses(neighbors)
        predictions.append(result)
        print('> predicted =' + repr(result) + ', actual=' + repr(test_set[x][-1]))
    # print(predictions)
    # print(test_set)
    accuracy = get_accuracy(test_set, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')


if __name__ == '__main__':
    main()
