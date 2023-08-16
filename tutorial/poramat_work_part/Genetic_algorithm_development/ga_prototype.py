import numpy as np
import matplotlib.pyplot as plt

class GA():
    
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
        # Initialize the Genetic Algorithm with the given parameters
        
        # Create a random population of individuals
        self.population = self.__create_random_population(dna_size, 
                                                          mutation_sigma, 
                                                          dna_start_position,
                                                          population_size)
        # Clip the population within the specified DNA bounds
        self.population = np.clip(self.population, dna_bounds[0], dna_bounds[1])
        
        # Initialize an array to store the fitness values of each individual
        self.fitnesses = np.zeros_like(self.population)
        
        # Initialize the best DNA and fitness values
        self.best_dna = None
        self.best_fitness = None
        
        # Store the DNA bounds, elitism rate, population size, mutation rate, mutation sigma,
        # mutation decay, mutation limit, mutation function, and crossover function
        self.dna_bounds = dna_bounds
        self.elitism = elitism
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.mutation_decay = mutation_decay
        self.mutation_limit = mutation_limit
        self.mutate_fn = mutate_fn
        self.crossover_fn = crossover_fn

        # Initialize a list to store the fitness values at each generation
        self.best_fitness_history = []
        self.average_fitness_history = []
                
    def get_solutions(self):
        # Update the mutation sigma using decay and limit it
        self.mutation_sigma *= self.mutation_decay
        self.mutation_sigma = np.maximum(self.mutation_sigma, self.mutation_limit)
        
        # Return the current population
        return self.population
        
        
    def set_fitnesses(self, fitnesses):
        # Set the fitness values for the current population
        
        assert len(fitnesses) == len(self.fitnesses)
        
        # Convert the fitness values to an array
        self.fitnesses = np.array(fitnesses)
        
        # Sort the fitness values and population based on the fitness values
        fitnesses_indices = self.fitnesses.argsort()
        sorted_fitnesses = self.fitnesses[fitnesses_indices]
        
        # Calculate the weighting of each individual based on its fitness
        fitnesses_weighting = np.maximum(0, 1 - sorted_fitnesses / self.fitnesses.sum())
        fitnesses_weighting /= fitnesses_weighting.sum()
        
        # Sort the population based on the fitness values
        sorted_population = self.population[fitnesses_indices]
        
        # Update the best DNA and fitness values
        self.best_dna = sorted_population[0]
        self.best_fitness = sorted_fitnesses[0]
        
        # Determine the number of new individuals to generate
        amount_new = int((1 - self.elitism) * len(self.population))
        
        # Generate new individuals through crossover and mutation
        new_population = []
        for _ in range(amount_new):
            # Select two parents based on fitness weighting
            i0 = np.random.choice(sorted_population.shape[0], p=fitnesses_weighting)
            i1 = np.random.choice(sorted_population.shape[0], p=fitnesses_weighting)
            
            # Create a new DNA by crossover and mutation
            new_dna = self.__crossover(self.population[i0], self.population[i1])            
            new_dna = self.__mutate(new_dna, self.mutation_sigma, self.mutation_rate)
            new_population.append(new_dna)

        amount_old = self.population_size - amount_new
        new_population = np.array(new_population + sorted_population[:amount_old].tolist())

        assert new_population.shape == self.population.shape
        
        self.population = np.clip(new_population, self.dna_bounds[0], self.dna_bounds[1])

        # Collect the best fitness value at each generation
        self.best_fitness_history.append(self.best_fitness)
        
        # Calculate and collect the average fitness value at each generation
        average_fitness = np.mean(sorted_fitnesses)
        self.average_fitness_history.append(average_fitness)

        
    def get_best(self):
        # Get the best DNA and fitness values
        
        return self.best_dna, self.best_fitness
        
        
    def __create_random_population(self, 
                                   dna_size, 
                                   dna_sigma, 
                                   dna_start_position,
                                   population_size):
        # Create a random population of individuals
        
        population = np.random.standard_normal((population_size, dna_size)) * dna_sigma
        population = 0.5 + 0.2 * (population / (2 * dna_sigma))
        return population + dna_start_position
    
    
    def __mutate(self, 
                 dna, 
                 mutation_sigma, 
                 mutation_rate):
        # Mutate the DNA of an individual
        
        if self.mutate_fn is not None:
            return self.mutate_fn(dna)
        
        if np.random.random_sample() < mutation_rate:
            dna += np.random.standard_normal(size=dna.shape) * mutation_sigma
        
        return dna
        
    
    def __crossover(self, dna1, dna2):
        # Perform crossover between two parent DNAs
        
        assert len(dna1) == len(dna2)
        
        if self.crossover_fn is not None:
            return self.crossover_fn(dna1, dna2)

        new_dna = np.copy(dna1)
        indices = np.where(np.random.randint(2, size=new_dna.size))
        new_dna[indices] = dna2[indices]
        return new_dna

    def plot_fitness(self):
        # Plot the best fitness and average fitness values against the generation
        generations = range(len(self.best_fitness_history))
        
        plt.plot(generations, self.best_fitness_history, label='Best Cost')
        plt.plot(generations, self.average_fitness_history, label='Average Cost')
        
        plt.xlabel('Generation')
        plt.ylabel('Cost')
        plt.title('Costs vs Generation')
        plt.legend()
        plt.show()