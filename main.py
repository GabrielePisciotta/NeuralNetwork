import numpy as np
from GridSearch import GridSearch
from KFoldCrossValidation import KFoldCrossValidation
from NeuralNetwork import NeuralNetwork
from Utilities import ReadMonk


def main(kind):
    train_examples, train_labels, test_examples, test_labels = ReadMonk(1)
    
    algo = 'minibatch'
    batchSize = len(train_examples)
    momentumAlpha = 0#.8
    momentumBeta = 0#.8

    losstype = 'squareloss'
    regularizationtype = 'l2'
    regularizationlambda = 0 #0.001

    learnRate = 0.00001#4.5 #0.2

    tr_mse = []
    ts_mse = []
    tr_acc = []
    ts_acc = []

    for i in range(1):
        n = NeuralNetwork(
                train_examples.values,
                train_labels.values,
                losstype=losstype,
                regtype=regularizationtype,
                reglambda=regularizationlambda,
                learning_rate=learnRate,
                momentumAlpha=momentumAlpha,
                momentumBeta=momentumBeta,
                algo=algo,
                batchSize=batchSize,
                restart=1,
                numHiddenLayers= 1,
                numOfUnitsPerLayer = 2,
                numOfUnitPerOutput = 1,
                weightsInitializer = 'xavier',
                task = 'classification',
                kind = kind,
                epochs=1000

        )


        tr_err, ts_err = n.train(plot=True, minimumVariation = 0.01, validation_set=test_examples.values, validation_labels=test_labels.values)
        tr_mse.append(tr_err[-1])
        ts_mse.append(ts_err[-1])
        tr_acc.append(n.test(train_examples.values, train_labels.values))
        ts_acc.append(n.test(test_examples.values, test_labels.values))

    print("(MSE) TR: {}, TS: {}".format(np.mean(tr_mse), np.mean(ts_mse)))
    print("(Accuracy) TR: {}, TS: {}".format(np.mean(tr_acc), np.mean(ts_acc)))

def example_gridsearch_kfold():
    print("Running Gridsearch + KFold")
    train_examples, train_labels, test_examples, test_labels = ReadMonk(1)

    kf = KFoldCrossValidation(3, train_examples.values, train_labels.values)
    
    _ALGORITHM = ['minibatch']
    _BATCH_SIZE = ['max']
    
    _RESTART_COUNT = [3]
    
    _HIDDEN_LAYERS_COUNT = [1]
    _UNITS_PER_HIDDEN_LAYER = [2, 3, 4]
    _UNITS_PER_OUTPUT_LAYER = [-1]    
    
    _LOSS_FUNCTION = ['squareloss']
    _LEARNING_RATE = list(np.arange(0.05, 0.5, 0.05))
    
    _REGULARIZATION_TYPE = ['l2']
    _REGULARIZATION_LAMBDA = [0, 0.0001, 0.0002]
    
    _MOMENTUM_ALPHA = list(np.arange(0, 0.99, 0.3))
    _MOMENTUM_BETA = [0]

    parameters_for_gridsearch = {
        'losstype': _LOSS_FUNCTION,
        'regtype': _REGULARIZATION_TYPE,
        'reglambda': _REGULARIZATION_LAMBDA,
        'learning_rate': _LEARNING_RATE,
        'momentumAlpha': _MOMENTUM_ALPHA,
        'momentumBeta': _MOMENTUM_BETA,
        'algo': _ALGORITHM,
        'batchSize': _BATCH_SIZE,
        'restart': _RESTART_COUNT,
        'numHiddenLayers': _HIDDEN_LAYERS_COUNT,
        'numOfUnitsPerLayer': _UNITS_PER_HIDDEN_LAYER,
        'numOfUnitPerOutput': _UNITS_PER_OUTPUT_LAYER
    }


    gs = GridSearch(parameters_for_gridsearch)

    for number in range(len(gs.get_grid())):
        message_for_pbar = "{} / {}".format(number+1, len(gs.get_grid()))
        kf.validate(gs.get_grid()[number], description_of_pbar=message_for_pbar, minimumVariation=0.01, )

def GridSearch_Batch_Monk1():
    print("Running Gridsearch + KFold")
    train_examples, train_labels, test_examples, test_labels = ReadMonk(1)

    kf = KFoldCrossValidation(3, train_examples.values, train_labels.values)
    
    _ALGORITHM = ['minibatch']
    _BATCH_SIZE = ['max']
    
    _RESTART_COUNT = [3]
    
    _HIDDEN_LAYERS_COUNT = [1]
    _UNITS_PER_HIDDEN_LAYER = [2, 3, 4]
    _UNITS_PER_OUTPUT_LAYER = [-1]    
    
    _LOSS_FUNCTION = ['squareloss']
    _LEARNING_RATE = list(np.arange(0.05, 0.5, 0.05))
    
    _REGULARIZATION_TYPE = ['l2']
    _REGULARIZATION_LAMBDA = [0, 0.0001, 0.0002]
    
    _MOMENTUM_ALPHA = list(np.arange(0, 0.99, 0.3))
    _MOMENTUM_BETA = [0]

    parameters_for_gridsearch = {
        'losstype': _LOSS_FUNCTION,
        'regtype': _REGULARIZATION_TYPE,
        'reglambda': _REGULARIZATION_LAMBDA,
        'learning_rate': _LEARNING_RATE,
        'momentumAlpha': _MOMENTUM_ALPHA,
        'momentumBeta': _MOMENTUM_BETA,
        'algo': _ALGORITHM,
        'batchSize': _BATCH_SIZE,
        'restart': _RESTART_COUNT,
        'numHiddenLayers': _HIDDEN_LAYERS_COUNT,
        'numOfUnitsPerLayer': _UNITS_PER_HIDDEN_LAYER,
        'numOfUnitPerOutput': _UNITS_PER_OUTPUT_LAYER
    }

    gs = GridSearch(parameters_for_gridsearch)

    for number in range(len(gs.get_grid())):
        message_for_pbar = "{} / {}".format(number+1, len(gs.get_grid()))
        kf.validate(gs.get_grid()[number], message_for_pbar, 0.01)

def GridSearch_MiniBatch_Monk1():
    print("Running Gridsearch + KFold")
    train_examples, train_labels, test_examples, test_labels = ReadMonk(1)

    kf = KFoldCrossValidation(3, train_examples.values, train_labels.values)
    
    _ALGORITHM = ['minibatch']
    _BATCH_SIZE = [5, 10, 20]
    
    _RESTART_COUNT = [2]
    
    _HIDDEN_LAYERS_COUNT = [1]
    _UNITS_PER_HIDDEN_LAYER = [3, 4]
    _UNITS_PER_OUTPUT_LAYER = [-1]    
    
    _LOSS_FUNCTION = ['squareloss']
    _LEARNING_RATE = list(np.arange(0.5, 2.2, 0.5)) + list(np.arange(3, 7, 1.5))
    
    _REGULARIZATION_TYPE = ['l2']
    _REGULARIZATION_LAMBDA = [0, 0.0001, 0.00001]
    
    _MOMENTUM_ALPHA = list(np.arange(0, 0.99, 0.4))
    _MOMENTUM_BETA = list(np.arange(0, 0.99, 0.2))

    parameters_for_gridsearch = {
        'losstype': _LOSS_FUNCTION,
        'regtype': _REGULARIZATION_TYPE,
        'reglambda': _REGULARIZATION_LAMBDA,
        'learning_rate': _LEARNING_RATE,
        'momentumAlpha': _MOMENTUM_ALPHA,
        'momentumBeta': _MOMENTUM_BETA,
        'algo': _ALGORITHM,
        'batchSize': _BATCH_SIZE,
        'restart': _RESTART_COUNT,
        'numHiddenLayers': _HIDDEN_LAYERS_COUNT,
        'numOfUnitsPerLayer': _UNITS_PER_HIDDEN_LAYER,
        'numOfUnitPerOutput': _UNITS_PER_OUTPUT_LAYER
    }

    gs = GridSearch(parameters_for_gridsearch)

    for number in range(len(gs.get_grid())):
        message_for_pbar = "{} / {}".format(number+1, len(gs.get_grid()))
        kf.validate(gs.get_grid()[number], message_for_pbar, 0.01)

def GridSearch_Monk2():
    print("Running Gridsearch + KFold")
    train_examples, train_labels, test_examples, test_labels = ReadMonk(2)

    kf = KFoldCrossValidation(3, train_examples.values, train_labels.values)
    
    _ALGORITHM = ['minibatch']
    _BATCH_SIZE = [20]
    
    _RESTART_COUNT = [2]
    
    _HIDDEN_LAYERS_COUNT = [1]
    _UNITS_PER_HIDDEN_LAYER = [4]
    _UNITS_PER_OUTPUT_LAYER = [-1]    
    
    _LOSS_FUNCTION = ['squareloss']
    _LEARNING_RATE = list(np.arange(0.1, 1, 0.2)) + list(np.arange(1, 4.2, 1)) 
    
    _REGULARIZATION_TYPE = ['l2']
    _REGULARIZATION_LAMBDA = [0, 0.00001, 0.0001]
    
    _MOMENTUM_ALPHA = [0] + list(np.arange(0.4, 1, 0.25))
    _MOMENTUM_BETA = list(np.arange(0.3, 1, 0.2))

    parameters_for_gridsearch = {
        'losstype': _LOSS_FUNCTION,
        'regtype': _REGULARIZATION_TYPE,
        'reglambda': _REGULARIZATION_LAMBDA,
        'learning_rate': _LEARNING_RATE,
        'momentumAlpha': _MOMENTUM_ALPHA,
        'momentumBeta': _MOMENTUM_BETA,
        'algo': _ALGORITHM,
        'batchSize': _BATCH_SIZE,
        'restart': _RESTART_COUNT,
        'numHiddenLayers': _HIDDEN_LAYERS_COUNT,
        'numOfUnitsPerLayer': _UNITS_PER_HIDDEN_LAYER,
        'numOfUnitPerOutput': _UNITS_PER_OUTPUT_LAYER
    }

    gs = GridSearch(parameters_for_gridsearch)

    for number in range(len(gs.get_grid())):
        message_for_pbar = "{} / {}".format(number+1, len(gs.get_grid()))
        kf.validate(gs.get_grid()[number], message_for_pbar, 0.01)
        
def GridSearch_Monk3():
    print("Running Gridsearch + KFold")
    train_examples, train_labels, test_examples, test_labels = ReadMonk(3)

    kf = KFoldCrossValidation(3, train_examples.values, train_labels.values)
    
    _ALGORITHM = ['minibatch']
    _BATCH_SIZE = [20]
    
    _RESTART_COUNT = [3]
    
    _HIDDEN_LAYERS_COUNT = [1]
    _UNITS_PER_HIDDEN_LAYER = [4, 5, 6]
    _UNITS_PER_OUTPUT_LAYER = [-1]    
    
    _LOSS_FUNCTION = ['squareloss']
    _LEARNING_RATE = [0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]
    
    
    _REGULARIZATION_TYPE = ['l2']
    _REGULARIZATION_LAMBDA = [0, 0.00001, 0.0001]
    
    _MOMENTUM_ALPHA = [0.6, 0.7, 0.8, 0.9]
    _MOMENTUM_BETA = [0.5, 0.6, 0.7, 0.8]

    parameters_for_gridsearch = {
        'losstype': _LOSS_FUNCTION,
        'regtype': _REGULARIZATION_TYPE,
        'reglambda': _REGULARIZATION_LAMBDA,
        'learning_rate': _LEARNING_RATE,
        'momentumAlpha': _MOMENTUM_ALPHA,
        'momentumBeta': _MOMENTUM_BETA,
        'algo': _ALGORITHM,
        'batchSize': _BATCH_SIZE,
        'restart': _RESTART_COUNT,
        'numHiddenLayers': _HIDDEN_LAYERS_COUNT,
        'numOfUnitsPerLayer': _UNITS_PER_HIDDEN_LAYER,
        'numOfUnitPerOutput': _UNITS_PER_OUTPUT_LAYER
    }

    gs = GridSearch(parameters_for_gridsearch)

    for number in range(len(gs.get_grid())):
        message_for_pbar = "{} / {}".format(number+1, len(gs.get_grid()))
        kf.validate(gs.get_grid()[number], message_for_pbar, 0.01)

if __name__ == '__main__':
    main(kind='L-BFGS')
    #main(kind='SGD')