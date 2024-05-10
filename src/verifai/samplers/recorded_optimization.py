
import pandas as pd
from verifai.samplers.domain_sampler import BoxSampler

class RecordedSampler(BoxSampler):
    def __init__(self, domain, params):
        super().__init__(domain)
        from numpy import zeros, ones

        self.rho = None

        filename = params['filename']
        self.df = pd.read_csv(filename)
        self.row = 0

        self.x = self.df.iloc[self.row]

    def getVector(self):
        return tuple(self.x), None

    def updateVector(self, vector, info, rho):
        self.rho = rho