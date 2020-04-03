import numpy as np 
import pandas as pd
from scipy.optimize import curve_fit

class Analytics():

    def __init__(self):
        super().__init__()

    def prob(self, dataset, max_bins=10):
        """Joint probability distribution P(X) for the given data."""

        # bin by the number of different values per feature
        num_rows, num_columns = dataset.shape
        bins = [min(len(np.unique(dataset[:, ci])), max_bins) for ci in range(num_columns)]

        freq, _ = np.histogramdd(dataset, bins)
        p = freq / np.sum(freq)
        return p
    
    def shannon_entropy(self, p):
        """Shannon entropy H(X) is the sum of P(X)log(P(X)) for probabilty distribution P(X)."""
        p = p.flatten()
        return -sum(pi*np.log2(pi) for pi in p if pi)

    def conditional_shannon_entropy(self, p, *conditional_indices):
        """Shannon entropy of P(X) conditional on variable j"""

        axis = tuple(i for i in np.arange(len(p.shape)) if i not in conditional_indices)

        return self.shannon_entropy(p) - self.shannon_entropy(np.sum(p, axis=axis))

    def mutual_information(self, p, j):
        """Mutual information between all variables and variable j"""
        return self.shannon_entropy(np.sum(p, axis=j)) - self.conditional_shannon_entropy(p, j)

    def conditional_mutual_information(self, p, j, *conditional_indices):
        """Mutual information between variables X and variable Y conditional on variable Z."""

        return self.conditional_shannon_entropy(np.sum(p, axis=j), *conditional_indices) - self.conditional_shannon_entropy(p, j, *conditional_indices)

