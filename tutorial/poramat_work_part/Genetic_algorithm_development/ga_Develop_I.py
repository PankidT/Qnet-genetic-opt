import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

class GA_Develop_I():

    def __init__(self, 
                 dna_size, 
                 dna_bounds, 
                 dna_start_position=None,
                 elitism=0.01,
                 population_size=200,
                 mutation_rate=0.01,
                 mutation_sigma=0.1,
                 mutation_decay=0.999,
                 mutation_limit=0.01,
                 mutate_fn=None,
                 crossover_fn=None):
        
        self.population = self.__create_random_population(dna_size, 
                                                          mutation_sigma, 
                                                          dna_start_position,
                                                          population_size)
        
        self.population = np.clip(self.population, dna_bounds[0], dna_bounds[1])

        self.cost = np.zeros_like(self.population)

        self.dna_bounds = dna_bounds
        self.elitism = elitism
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.mutation_decay = mutation_decay
        self.mutation_limit = mutation_limit
        self.mutate_fn = mutate_fn
        self.crossover_fn = crossover_fn

        # Initialize the best DNA and cost values
        self.best_dna = None
        self.lowest_cost = None

        # Initialize a list to store the cost values at each generation
        self.lowest_cost_history = []
        self.average_cost_history = []

    # def __create_random_population(
    #     self,
    #     dna_size, 
    #     dna_sigma, 
    #     dna_start_position,
    #     population_size):
    #     # Create a random population of individuals
        
    #     # (dimentsion of array)
    #     population = np.random.standard_normal((population_size, dna_size)) * dna_sigma
        
    #     return population + dna_start_position
    
    def __create_random_population(
        self,
        dna_size, 
        dna_sigma, 
        dna_start_position,
        population_size):
        population = np.random.standard_normal((population_size, dna_size)) * dna_sigma
        shifted_population = 0.5 + 0.3 * (population / (2 * dna_sigma))
        return shifted_population + dna_start_position
    
    def __crossover(self, dna1, dna2):

        if self.crossover_fn is not None:
            return self.crossover_fn(dna1, dna2)

        assert len(dna1) == len(dna2)

        new_dna = np.copy(dna1)
        indices = np.where(np.random.randint(2, size=new_dna.size))
        new_dna[indices] = dna2[indices]
        return new_dna

    def __mutate(self, dna, mutation_sigma, mutation_rate):

        if self.mutate_fn is not None:
            return self.mutate_fn(dna, mutation_sigma, mutation_rate)

        if np.random.random_sample() < mutation_rate:
            dna += np.random.standard_normal(size=dna.shape) * mutation_sigma
            # np.add(dna, np.random.standard_normal(size=dna.shape) * mutation_sigma, out=dna, casting="unsafe")

        return dna
    
    def evolve(self, cost_array):

        assert len(cost_array) == self.population_size

        self.cost = np.array(cost_array)

        cost_indices = self.cost.argsort()
        sorted_cost_array = self.cost[cost_indices]

        cost_weight = np.maximum(0, 1 - sorted_cost_array / self.cost.sum())
        cost_weight /= cost_weight.sum()

        sorted_population = self.population[cost_indices]        

        self.best_dna = sorted_population[0]
        self.lowest_cost = sorted_cost_array[0]
        
        # Calculate the number of individuals to keep for the next generation
        amount_new = int(self.population_size * self.elitism)        

        new_population = []
        for i in range(amount_new):
            # choose two individual from population
            index_ind_0 = np.random.choice(sorted_population.shape[0], p=cost_weight)
            index_ind_1 = np.random.choice(sorted_population.shape[0], p=cost_weight)

            new_dna = self.__crossover(sorted_population[index_ind_0], sorted_population[index_ind_1])
            new_dna = self.__mutate(new_dna, self.mutation_sigma, self.mutation_rate)
            new_population.append(new_dna)

        amount_old = self.population_size - amount_new
        new_population = np.array(new_population + sorted_population[:amount_old].tolist())

        assert new_population.shape == self.population.shape
        
        self.population = np.clip(new_population, self.dna_bounds[0], self.dna_bounds[1])

        # Collect the best cost value at each generation
        self.lowest_cost_history.append(self.lowest_cost)

        # Calculate and collect the average cost value at each generation
        average_cost = np.mean(sorted_cost_array)
        self.average_cost_history.append(average_cost)

        return self.best_dna, self.lowest_cost

    def plot(self):
        plt.plot(self.lowest_cost_history, label="Lowest cost")
        plt.plot(self.average_cost_history, label="Average cost")
        plt.legend()
        plt.ylabel("Cost")
        plt.xlabel("Generation")
        plt.show()

import datetime

class ExperimentResult():
    def __init__(self, experiment_name, weight1, weight2, mutation_rate,
                 population_size, elitism, amount_optimization_steps, ga_object=None):
        self.experiment_config = {
            'Experiment_name': experiment_name,
            'DateCreate': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Hyperparameters': {
                'w1': weight1,
                'w2': weight2,
                'mutation_rate': mutation_rate,
                'num_population': population_size,
                'num_parents': int(population_size * elitism),
                'num_generation': amount_optimization_steps,
            },
            'Parameter_history': {
                'loss': [],
                'gate_error': [],
                'measurement_error': [],
                'memory_time': []
            },
            'fidelity_history': [],
            'cost_history': [],            
            'ga_object': ga_object
        }

    def add_ga_object(self, ga_object):
        self.experiment_config['ga_object'] = ga_object

    def save(self, file_name, save_config=True, folder_name='results'):

        if save_config:
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

        with open(file_name, 'wb') as f:
            pickle.dump(self, f)