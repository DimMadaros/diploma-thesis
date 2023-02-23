"""
Created on Mon Jan 16 12:11:54 2023

@author: Dimitris Madaros
"""
import time
import numpy as np
from numpy.random import default_rng


class GA:
    def __init__(self, vrp, max_iter=1000):
        self.vrp = vrp  # An instance of a VRP problem
        self.max_iter = max_iter  # Max number of iterations
        self.chromosomes = 100  # Number of chromosomes
        self.genes = self.vrp.size  # The number of genes of the chromosomes
