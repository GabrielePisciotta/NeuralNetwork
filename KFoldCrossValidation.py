from sklearn.utils import shuffle
from math import sqrt
import numpy as np
from multiprocessing import Pool
from queue import Queue
from tqdm import *
from scipy.stats import kurtosis
from timeit import default_timer as timer
from Utilities import printToFile
from NeuralNetwork import NeuralNetwork
RandomState = 4200000

class KFoldCrossValidation:
    def __init__(self, k, dataset, labels, name=""):
        self.k = k
        self.name = name
        dataset, labels = shuffle(dataset, labels, random_state=RandomState)
        self.split(dataset, labels)

    def split(self, dataset, labels):
        self.groups = []
        self.groups_labels = []

        actual_group = []
        actual_labels = []
        
        (quo, rem) = divmod(len(dataset), self.k)
        
        for i in range(self.k - 1):
            start_pos = i * quo
            end_pos = (i+1)*quo
            
            actual_group = dataset[start_pos : end_pos]
            actual_labels = labels[start_pos : end_pos]
            
            self.groups.append(np.array(actual_group))
            self.groups_labels.append(np.array(actual_labels))
        
        # Last Group
        actual_group = dataset[(self.k - 1) * quo : ]
        actual_labels = labels[(self.k - 1) * quo : ]
            
        self.groups.append(np.array(actual_group))
        self.groups_labels.append(np.array(actual_labels))

    def get_train_and_test(self, index):
        training_set = []
        training_labels = []
        
        for i in range(len(self.groups)):
            if i != index:
                for j in range(len(self.groups[i])):
                    training_set.append(self.groups[i][j])
                    training_labels.append(self.groups_labels[i][j])

        test_set = self.groups[index]
        test_labels = self.groups_labels[index]
        return np.array(training_set), np.array(training_labels), test_set, test_labels

    def worker(self, data):
        params, plot, index, minimumVariation = data
        training_set, training_labels, test_set, test_labels = self.get_train_and_test(index)

        if (params['batchSize'] == 'max'):
            params.update({'batchSize': len(training_set)})
        params.update({'training_set': training_set, 'training_labels': training_labels})
        
        n = NeuralNetwork(**params)
        n.train(plot, minimumVariation, test_set, test_labels)

        return n.test(test_set, test_labels)

    def validate(self, params, minimumVariation, description_of_pbar = "", plot=False):

        start = timer()
        index = [(params, plot, x, minimumVariation) for x in range(self.k)]

        with Pool() as p:
            values = list(tqdm(p.imap_unordered(self.worker, index), total = self.k))#, desc = description_of_pbar))

        mean = np.mean(values)
        std = np.std(values)
        
        end = timer()
        time_elapsed = end - start

        params.update({'mean': mean, 'std':std, 'time elapsed':time_elapsed})

        printToFile(params,"grid_search_results_{}.csv".format(self.name))


