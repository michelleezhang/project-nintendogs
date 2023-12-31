import cma
import numpy as np
import time
import torch
from matplotlib import pyplot as plt
from get_level import GetLevel

# another optimization strategy, CMA-ES
class SimpleCMAES:
    '''
    Class for CMA-ES, an optimization strategy.
    Returns best solution and plots results
    '''
    def __init__(self, standard_deviation, population_size, noise_size, mean=None):
        self.population_size = population_size
        self.sigma = standard_deviation
        self.mean = (
            mean if mean is not None else np.random.random_sample(noise_size).flatten()
        ) # generate mean vector by a random_sample where the size is a flatten(14*14)
        self.populations = []
        self.best_solution = self.mean
        self.reset()

    def reset(self, mean=None):
        mean = mean if mean is not None else self.mean
        self.es = cma.CMAEvolutionStrategy(
            mean, self.sigma, {"popsize": self.population_size, "bounds": [-1, 1]}
        )
    
    def get_pop_noise(self):
        '''
        Return newly generated candidate solutions
        '''
        #get new candidate solutions, sampled from a multi-variate normal distribution and transformed to f-representation (phenotype) to be evaluated.
        self.populations = self.es.ask()
        
        # print(self.populations)
        # print("in get noise")
        # tmp = input()
        # print("in get noise")
        return self.populations

    def update_cmaes(self, fitnesses):
        '''
        Updates with new best solution
        '''
        self.es.tell(self.populations, fitnesses)
        self.es.logger.add()
        self.best_solution = self.es.best.x

    def get_best_solution(self):
        return self.best_solution
    
    def plot_cma(self, dir_path=None, count=0):
        '''
        Plot cma-es values
        '''
        self.es.logger.plot()
        # cma.plot()
        plt.savefig(dir_path + "CMAES_" + str(count) + ".png")
        # cma.s.figshow()
        # cma.s.figsave(str(count)+'.png')
        # self.es.s.figclose()
