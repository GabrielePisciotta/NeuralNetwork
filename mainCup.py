import numpy as np
from GridSearch import GridSearch
from KFoldCrossValidation import KFoldCrossValidation
from NeuralNetwork import NeuralNetwork
from Utilities import ReadCup, saveResultsAsCsv, ReadBlindCup


# This script is for the CUP Challlenge
# of the ML Course 19-20 @ University Of Pisa
# -- dataset not included --
def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = ReadCup()

    real_test_set = ReadBlindCup()
    algo = 'minibatch'
    batchSize = 100
    momentumAlpha = 0.9
    momentumBeta = 0.8

    losstype = 'squareloss'
    regularizationtype = 'l2'
    regularizationlambda = 0.0000  #0.001

    learnRate = 0.001

    X_train = np.concatenate([X_train, X_valid])
    y_train = np.concatenate([y_train, y_valid])

    n = NeuralNetwork(
            X_train,
            y_train,
            losstype=losstype,
            regtype=regularizationtype,
            reglambda=regularizationlambda,
            learning_rate=learnRate,
            momentumAlpha=momentumAlpha,
            momentumBeta=momentumBeta,
            algo=algo,
            batchSize=batchSize,
            restart=1,
            numHiddenLayers= 3,
            numOfUnitsPerLayer = 30,
            numOfUnitPerOutput = 2,
            weightsInitializer = 'xavier',
            task = 'regression'
    )


    n.train(plot=True,
            minimumVariation=0.75,
            validation_set=X_valid,
            validation_labels=y_valid
            )

    print("MEE on test: {}".format(n.test(X_test, y_test)))

    predictions = n.predict(real_test_set.values)
    saveResultsAsCsv(predictions)

def GridSearch_Cup():
    print("Running Gridsearch + KFold")
    X_train, X_valid, X_test, y_train, y_valid, y_test = ReadCup()
    KFoldCount = 10

    training_set = np.concatenate([X_train, X_valid])
    training_labels = np.concatenate([y_train, y_valid])

    kf = KFoldCrossValidation(KFoldCount, training_set, training_labels)
    
    _ALGORITHM = ['minibatch']
    _BATCH_SIZE = [100]
    
    _RESTART_COUNT = [1]
    
    _HIDDEN_LAYERS_COUNT = [1, 3, 5]
    _UNITS_PER_HIDDEN_LAYER = [10, 20, 30]
    _UNITS_PER_OUTPUT_LAYER = [2]    
    
    _LOSS_FUNCTION = ['squareloss']
    _LEARNING_RATE = [0.0015, 0.001, 0.0005]
    
    _REGULARIZATION_TYPE = ['l2']
    _REGULARIZATION_LAMBDA = [0, 0.00001]
    
    _MOMENTUM_ALPHA = [0.5, 0.7, 0.9]
    _MOMENTUM_BETA = [0.4, 0.6, 0.8]
    _TASK = ["regression"]

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
        'numOfUnitPerOutput': _UNITS_PER_OUTPUT_LAYER,
        'task': _TASK
    }

    gs = GridSearch(parameters_for_gridsearch)

    for number in range(len(gs.get_grid())):
        message_for_pbar = "{} / {}".format(number+1, len(gs.get_grid()))
        kf.validate(gs.get_grid()[number], minimumVariation=0.8, description_of_pbar=message_for_pbar, plot=False)

if __name__ == '__main__':
    main()
