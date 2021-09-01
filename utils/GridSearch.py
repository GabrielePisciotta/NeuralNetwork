import itertools
class GridSearch:
    def __init__(self, params):
        self.params = params
        self.grid = []
        self.generate_grid()

    def generate_grid(self):
        key, values = zip(*self.params.items())
        it = itertools.product(*values)

        for elements in it:
            parameters = dict(zip(key, elements))
            if parameters['batchSize'] != 'max' and parameters['momentumBeta'] == 0:
                continue
            self.grid.append(parameters)

    def get_grid(self):
        return self.grid
