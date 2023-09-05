import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import datetime
import seaborn as sns
import pandas as pd

sys.path.insert(0, '/Users/poramat/Documents/qwanta/tutorial/poramat_work_part')
from all_function import *

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
    
    def __create_random_population(
        self,
        dna_size, 
        dna_sigma, 
        dna_start_position,
        population_size):
        population = np.random.standard_normal((population_size, dna_size)) * dna_sigma
        shifted_population = 0.5 + 0.25 * (population / (2 * dna_sigma))
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
        plt.plot(self.lowest_cost_history, label="Lowest Cost")
        plt.plot(self.average_cost_history, label="Average Cost")
        plt.xlabel("Generation")
        plt.ylabel("Cost")
        plt.title("Cost over Generations")
        plt.legend()
        plt.show()

class ExperimentResult():
    def __init__(self, 
            experiment_name, 
            strategy, 
            weight1, 
            weight2, 
            mutation_rate,
            population_size, 
            elitism, 
            amount_optimization_steps, 
            ga_object=None):
        self.experiment_config = {
            'Experiment_name': experiment_name,            
            'strategy': strategy,
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
            'fidelity_history': {
                'max': [],
                'mean': [],
                'min': []
            },
            'cost_history': {
                'max': [],
                'mean': [],
                'min': []
            },            
            'ga_object': ga_object
        }

    def add_ga_object(self, ga_object):
        self.experiment_config['ga_object'] = ga_object

    def add_parameter(self, parameter_name, value):
        self.experiment_config[parameter_name] = value

    def save(self, file_path, folder_name):

        folder_name = f'results_{folder_name}'        

        with open(file_path, 'wb') as f:
            pickle.dump(self, f)    

    def plot(self):
        sns.set_theme(style="darkgrid")

        x = np.arange(0, self.experiment_config['Hyperparameters']['num_generation'], 1)
        w1 = self.experiment_config['Hyperparameters']['w1']
        w2 = self.experiment_config['Hyperparameters']['w2']
        mutation_rate = self.experiment_config['Hyperparameters']['mutation_rate']        
        num_population = self.experiment_config['Hyperparameters']['num_population']
        num_parents = self.experiment_config['Hyperparameters']['num_parents']
        num_generation = self.experiment_config['Hyperparameters']['num_generation']

        max_fielity = self.experiment_config['fidelity_history']['max']
        max_cost = self.experiment_config['cost_history']['max']

        mean_fidelity = self.experiment_config['fidelity_history']['mean']
        mean_cost = self.experiment_config['cost_history']['mean']

        min_fidelity = self.experiment_config['fidelity_history']['min']
        min_cost = self.experiment_config['cost_history']['min']

        fig, ax = plt.subplots(2, 2, figsize=(15, 10))

        ax[0, 0].set_title(f'w1: {w1}, w2: {w2}, mr: {mutation_rate}, pop: {num_population}, parents: {num_parents}, ng: {num_generation}', loc='right')

        color = 'tab:red'
        ax[0, 0].set_xlabel('generation')
        ax[0, 0].set_ylabel('fidelity', color=color)
        ax[0, 0].plot(x, mean_fidelity, color=color)
        ax[0, 0].tick_params(axis='y', labelcolor=color)

        ax2 = ax[0, 0].twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('cost', color=color)  # we already handled the x-label with ax1
        ax2.plot(x, mean_cost, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        ax[0, 1].plot(mean_fidelity, mean_cost, '.')
        ax[0, 1].set_xlabel('fidelity')
        ax[0, 1].set_ylabel('cost')

        ax[1, 0].set_title('Fidelity of each generation')
        ax[1, 0].set_xlabel('generation')
        ax[1, 0].set_ylabel('fidelity')
        ax[1, 0].plot(x, max_fielity, label='max fidelity', color='tab:red')
        ax[1, 0].plot(x, min_fidelity, label='min fidelity', color='tab:green')
        ax[1, 0].plot(x, mean_fidelity, label='mean fidelity', color='tab:blue')        
        ax[1, 0].fill_between(x, max_fielity, min_fidelity, alpha=0.2, label='range fidelity')
        ax[1, 0].legend()

        ax[1, 1].set_title('Evolution of cost')
        color = 'tab:red'
        ax[1, 1].set_xlabel('generation')
        ax[1, 1].set_ylabel('avg cost', color=color)
        ax[1, 1].plot(x, mean_cost, 'r.-', label='mean cost', color=color)
        ax[1, 1].tick_params(axis='y', labelcolor=color)

        ax2 = ax[1, 1].twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('best cost', color=color)  # we already handled the x-label with ax1
        ax[1, 1].plot(x, min_cost, 'b+',label='min cost', color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        ax[1, 1].legend()

    def table(self):

        index_best_fidelity = np.argmax(self.experiment_config['fidelity_history']['max'])
        best_fidelity = self.experiment_config['fidelity_history']['max'][index_best_fidelity]

        baseline_parameter = parameterTransform(self.experiment_config['Parameter_history']['loss'][0],
                           self.experiment_config['Parameter_history']['memory_time'][0],
                           self.experiment_config['Parameter_history']['gate_error'][0],
                           self.experiment_config['Parameter_history']['measurement_error'][0],
                           )
        
        best_parameter = parameterTransform(self.experiment_config['Parameter_history']['loss'][index_best_fidelity],
                           self.experiment_config['Parameter_history']['memory_time'][index_best_fidelity],
                           self.experiment_config['Parameter_history']['gate_error'][index_best_fidelity],
                           self.experiment_config['Parameter_history']['measurement_error'][index_best_fidelity],
                           )

        data = {
            "loss (dB/km)": [baseline_parameter[0], best_parameter[0]],
            "memory_time (second)": [baseline_parameter[1], best_parameter[1]],
            "gate_error (%)": [baseline_parameter[2], best_parameter[2]],
            "measurement_error (%)": [baseline_parameter[3], best_parameter[3]],
        }

        row = ["Baseline", "Solution"]

        df = pd.DataFrame(data, index=row)
        print(f'Best fidelity: {best_fidelity}')

        return df
        