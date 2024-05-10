
import pandas as pd
from verifai.samplers.domain_sampler import BoxSampler

class RecordedSampler(BoxSampler):
    def __init__(self, domain, params):
        super().__init__(domain)
        from numpy import zeros, ones

        self.rho = None

        self.row = 0
        self.x = None

    def setFilename(self, filename):
        self.df = pd.read_csv(filename)
        self.x = tuple(list(self.df.iloc[self.row]))
        self.row = self.row + 1

    def getVector(self):
        if self.row < len(self.df.index):
            self.x = tuple(list(self.df.iloc[self.row]))
            self.row = self.row + 1
        return self.x, None

    def updateVector(self, vector, info, rho):
        self.rho = rho