#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


def euclidean_distance(a, b):
    """
    Returns a scalar value of the euclidean distance between
    two n-dimensional points.
    """
    diff = a - b
    return np.sqrt(np.dot(diff, diff))


def load_data(csv_filename):
    """ 
    Returns a numpy ndarray in which each row repersents
    a wine and each column represents a measurement. There should be 11
    columns (the "quality" column cannot be used for classificaiton).
    """

    # Our final data table will have 11 columns of type float.
    data_table = np.zeros((1, 11), dtype=float)

    with open(csv_filename, 'r') as file:

        # Skip heading row
        next(file)

        for line in file:
            # Split csv by commas
            row = line.split(';')
            # Remove last column
            row = row[:11]

            # The split method returns an array of strings. We want an array
            # of floats, so we must perform explicit typecasting.
            float_row = np.zeros((1, 11), dtype=float)
            for i in range(11):
                float_row[0][i] = float(row[i])

            # Append new row to the bottom of the data table
            data_table = np.vstack((data_table, float_row))

        # Since we began with a row of zeroes for our initial data_table,
        # slice our table to exclude the first row.
        data_table = data_table[1:]

    return data_table


def split_data(dataset, ratio=0.9):
    """
    Return a (train, test) tuple of numpy ndarrays. 
    The ratio parameter determines how much of the data should be used for 
    training. For example, 0.9 means that the training portion should contain
    90% of the data. You do not have to randomize the rows. Make sure that 
    there is no overlap. 
    """
    training_set = np.zeros((1, 11), dtype=float)
    test_set = np.zeros((1, 11), dtype=float)

    rows, columns = dataset.shape

    # This is the row number we will use to calculate the ratio.
    row_num = 0

    for row in dataset:

        row_num += 1

        # Separate training set and test set based on ratio
        if (row_num < ratio * rows):
            training_set = np.vstack((training_set, row))

        else:
            test_set = np.vstack((test_set, row))

    # Remove first row of zeroes
    training_set = training_set[1:, :]
    test_set = test_set[1:, :]

    return (training_set, test_set)


def compute_centroid(data):
    """
    Returns a 1D array (a vector), representing the centroid of the data
    set. 
    """

    # ndarray of floats to return
    result = np.zeros((11), dtype=float)
    num_rows = 0

    # Add all values together
    for row in data:
        result = np.add(result, row)
        num_rows += 1

    # Divide all values by number of entries and return
    return (result / num_rows)


def experiment(ww_train, rw_train, ww_test, rw_test):
    """
    Train a model on the training data by creating a centroid for each class.
    Then test the model on the test data. Prints the number of total 
    predictions and correct predictions. Returns the accuracy. 
    """

    # Create centroids from training sets
    ww_ctr = compute_centroid(ww_train)
    rw_ctr = compute_centroid(rw_train)

    num_correct = 0
    num_test = 0

    # Iterate over all red wine test elements
    for test_row in rw_test:

        num_test += 1

        # If red wine centroid is closer, prediction is correct.
        if euclidean_distance(test_row, ww_ctr) >  \
                euclidean_distance(test_row, rw_ctr):
            num_correct += 1

    # Iterate over all white wine test elements
    for test_row in ww_test:

        num_test += 1

        # If white wine centroid is closer, prediction is correct.
        if euclidean_distance(test_row, rw_ctr) > \
                euclidean_distance(test_row, ww_ctr):
            num_correct += 1

    # Compute accuracy, then print results.
    if num_test != 0:
        accuracy = num_correct / num_test
    else:
        return 0.0

    print("Number of predictions: {}".format(num_test))
    print("Number of correct predictions: {}".format(num_correct))
    print("Accuracy: {}".format(accuracy))

    return accuracy


def cross_validation(ww_data, rw_data, k):
    """
    Perform k-fold crossvalidation on the data and print the accuracy for each
    fold. 
    """

    i = 0
    total_accuracy = 0

    # The ww and rw data sets both have the same number of elements.
    num_rows = ww_data.shape[0]

    # The number of elements that should be in each k-partition
    divisor = (num_rows / k)

    for i in range(k):

        # The ndarrays that we will use as training and testing sets.
        ww_train = np.zeros((1, 11), dtype=float)
        rw_train = np.zeros((1, 11), dtype=float)
        ww_test = np.zeros((1, 11), dtype=float)
        rw_test = np.zeros((1, 11), dtype=float)

        # Iterate over all elements in the datasets.
        for j in range(num_rows):

            # If the current row is within the k-partition, add to test set.

            if i * divisor <= j and j <= (i + 1) * divisor:
                ww_test = np.vstack((ww_test, ww_data[j]))
                rw_test = np.vstack((rw_test, rw_data[j]))
                j += 1

            # Else, add to training set.
            else:
                ww_train = np.vstack((ww_train, ww_data[j]))
                rw_train = np.vstack((rw_train, rw_data[j]))

        # Remove initial row of zeros we started with.
        ww_train = ww_train[1:, :]
        rw_test = rw_test[1:, :]
        ww_train = ww_train[1:, :]
        rw_test = rw_test[1:, :]

        # Perform the experiment with this k-partition, then add to total acc.
        total_accuracy += experiment(ww_train, rw_train, ww_test, rw_test)

    # Return total accuracy divided by number of crossfold experiments.
    return (total_accuracy / k)


if __name__ == "__main__":

    ww_data = load_data('whitewine.csv')
    rw_data = load_data('redwine.csv')

    ww_train, ww_test = split_data(ww_data, 0.9)
    rw_train, rw_test = split_data(rw_data, 0.9)
    #experiment(ww_train, rw_train, ww_test, rw_test)

    # Uncomment the following lines for step 3:
    k = 8
    acc = cross_validation(ww_data, rw_data, k)
    print("{}-fold cross-validation accuracy: {}".format(k, acc))
