import numpy as np

class Accuracy():

    def __call__(self, test_set, labels, n):
        correctly_classified = 0
        for sample, label in zip(test_set, labels):
            if label >= 0.5:
                label = 1
            else:
                label = 0

            if n.predict(sample) >= 0.5:
                prediction = 1
            else:
                prediction = 0

            if (label == prediction):
                correctly_classified += 1

        return correctly_classified/len(test_set)*100