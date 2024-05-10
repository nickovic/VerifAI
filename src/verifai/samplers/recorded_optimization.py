
import pandas as pd
import numpy as np
from verifai.samplers.domain_sampler import BoxSampler

class RecordedSampler(BoxSampler):
    def __init__(self, domain, params):
        super().__init__(domain)

        self.lb = [domain.domains[idx].intervals[0][0] for idx in range(domain.flattenedDimension)]
        self.ub = [domain.domains[idx].intervals[0][1] for idx in range(domain.flattenedDimension)]

        self.rho = None

        self.row = 0
        self.x = None

    def setFilename(self, filename):
        self.df = pd.read_csv(filename)
        x = np.array(list(self.df.iloc[self.row]))
        lb = np.array(self.lb)
        ub = np.array(self.ub)
        x = (x - lb)/(ub - lb)
        self.x = tuple(list(x))
        self.row = self.row + 1

    def getVector(self):
        if self.row < len(self.df.index):
            x = np.array(list(self.df.iloc[self.row]))
            lb = np.array(self.lb)
            ub = np.array(self.ub)
            x = (x - lb) / (ub - lb)
            self.x = tuple(list(x))
            self.row = self.row + 1
        return self.x, None

    def updateVector(self, vector, info, rho):
        self.rho = rho