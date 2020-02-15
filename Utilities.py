import csv
import math
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils import shuffle


def rescale(labels, neg):  # Rescaling from 0-1 to (neg) - (1 - neg)
    mul = 1 - (2 * neg)
    labels = 2 * labels
    labels = labels - 1

    labels = mul * labels

    labels = labels + 1
    labels = labels / 2

    return labels


def OneOfKEncode(df):
    newdf = df.applymap(str)
    newdf = pd.get_dummies(newdf, prefix=df.columns)
    newdf = newdf.applymap(float)

    return newdf

def ReadMonk(n):
    # Negative label, used for rescaling in binary classification
    neg_label = 0.1
    pathTrain = 'datasets/monk{}/train'.format(n)
    pathTest = 'datasets/monk{}/train'.format(n)
    
    # Read dataset
    train_examples = pd.read_csv(pathTrain, sep=" ", usecols=[2, 3, 4, 5, 6, 7])
    train_examples = OneOfKEncode(train_examples)
    train_labels = pd.read_csv(pathTrain, sep=" ", usecols=[1])
    train_labels = rescale(train_labels, neg_label)

    # Read dataset
    test_examples = pd.read_csv(pathTest, sep=" ", usecols=[2, 3, 4, 5, 6, 7])
    test_examples = OneOfKEncode(test_examples)
    test_labels = pd.read_csv(pathTest, sep=" ", usecols=[1])
    
    return train_examples, train_labels, test_examples, test_labels

def printToFile(content, filename='grid_search_results.csv'):
    with open(filename, 'a') as outfile:
        listWriter = csv.DictWriter(
            outfile,
            fieldnames=content.keys(),
            delimiter=';',
            quotechar='|',
            quoting=csv.QUOTE_MINIMAL
        )
        if outfile.tell() == 0:
            listWriter.writeheader()
        listWriter.writerow(content)


def ReadCup():
    randomSeed = 420000
    designPerc = 0.8 # Percentuale del design set
    trainingPerc = 0.8 # Percentuale del training set (wrt design set)
    dataPath = 'datasets/cup/ML-CUP19-TR.csv'

    # Read dataset
    X = pd.read_csv(dataPath, skiprows=7, header=None, usecols=list(range(1, 21)))
    y = pd.read_csv(dataPath, skiprows=7, header=None, usecols=[21, 22])

    X = np.array(X)
    y = np.array(y)

    X, y = shuffle(X, y, random_state=randomSeed)

    designSize = math.floor(X.shape[0] * designPerc)
    X_design, X_test = X[: designSize], X[designSize : ]
    y_design, y_test = y[: designSize], y[designSize  :]
    
    trainingSize = math.floor(X_design.shape[0] * trainingPerc)
    X_train, X_valid = X_design[: trainingSize], X_design[trainingSize : ]
    y_train, y_valid = y_design[: trainingSize], y_design[trainingSize  :]

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def ReadCupWithoutSplit():
    dataPath = 'datasets/cup/ML-CUP19-TR.csv'
    testPath = 'datasets/cup/ML-CUP19-TS.csv'

    # Read dataset
    training_set = pd.read_csv(dataPath, skiprows=7, header=None, usecols=list(range(1, 21)))
    training_labels = pd.read_csv(dataPath, skiprows=7, header=None, usecols=[21, 22])
    test_set = pd.read_csv(testPath, skiprows=7, header=None)

    return np.array(training_set), np.array(training_labels.values), np.array(test_set)

def ReadBlindCup():
    dataPath = 'datasets/cup/ML-CUP19-TS.csv'

    # Read dataset
    df = pd.read_csv(dataPath, skiprows=7, index_col=0, header=None)

    return df

def plot_error_curve(epochs, error, error_on_validation):
    now = datetime.now()
    print("Number of epochs: {}".format(epochs))
    plt.plot(range(len(error)), error, "-b", label='Error on Training Set')
    plt.plot(range(len(error_on_validation)), error_on_validation, "--r", label='Error on Test Set')
    plt.legend(loc="upper right")
    plt.xlabel("Epoch")
    plt.ylabel("Error")

    plt.savefig('training_error{}.png'.format(now.strftime("%H%M%S")), format='png', dpi=300)
    plt.show()

def plot_accuracy_mee(accuracy_mee, accuracy_mee_on_training, label):
    now = datetime.now()
    plt.plot(range(len(accuracy_mee)), accuracy_mee, "--r", label=label+" on Test Set")
    plt.plot(range(len(accuracy_mee_on_training)), accuracy_mee_on_training, "blue", label=label+" on Training Set")

    plt.legend(loc="upper left")
    plt.xlabel("Epoch")
    plt.ylabel(label)

    plt.savefig('accuracy_mee{}.png'.format(now.strftime("%H%M%S")), format='png', dpi=300)
    plt.show()

def saveResultsAsCsv(content):

    with open("result.csv", 'w') as outfile:
        listWriter = csv.writer(
            outfile,
            delimiter=',',
            quotechar='',
            quoting=csv.QUOTE_NONE,
            escapechar='\\'
        )

        for i in range(len(content)):
            s = [i+1]
            for j in range(len(content[i])):
                s.append(content[i][j])
            listWriter.writerow(s)

