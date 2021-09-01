import numpy as np

class MEE():
    def __call__(self, test_set, labels, n):
        sommatoria = 0

        for sample, label in zip(test_set, labels):
            label = label.T
            prediction = n.predict(sample)
            prediction = prediction[0]
            sommatoria += np.linalg.norm(prediction-label)

        return sommatoria/len(test_set)