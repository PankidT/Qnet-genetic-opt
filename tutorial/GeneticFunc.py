import random
import matplotlib.pyplot as plt
import numpy as np
from math import log
import ray
from tqdm import tqdm

import sys
# from qwanta import Experiment


# print python version

print(sys.version)

class ExperimentHistory:
    def __init__(self):
        self.populationHistory = []
        self.costHistory = []

class CreateGeneticExperiment:
    def __init__(self, population_size: int=10, numParameter: int=4, parents_size: int=5, 
                 mutation_rate: float=0.1, generations: int=100, w1: float=0.5, w2: float=0.5,
                 Fidelity: float=0.0, minFidelity: float=0.0):
        self.POPULATION_SIZE = population_size
        self.NUM_PARAMETER = numParameter
        self.PARENTS_SIZE = parents_size
        self.MUTATION_RATE = mutation_rate
        self.GENERATIONS = generations
        # self.MIN_RATE = min_rate
        self.F = Fidelity
        self.MINFIDELITY = minFidelity

        self.population = []
        self.cost = []
        self.w1 = w1
        self.w2 = w2

    def create_population(self):
        # create individual in population
        population = []
        for i in range(self.POPULATION_SIZE):
            individual = [random.random() for i in range(self.NUM_PARAMETER)]
            population.append(individual)
        self.population = population
        return population
    
    def Heaviside(self, x: float):
        if x >= 0:
            return 1
        else:
            return 0
        
    def singleObject_cost(self, x: list, xNew: list, w1: float, w2: float):
        # cost function
        output = []
        for i in range(len(x)):
            k = log(xNew[i])/log(x[i])
            output.append(k)

        cost = sum(output)

        # Rmin = self.MIN_RATE
        cost_single_objective = w1*self.Heaviside(self.MINFIDELITY-self.F) + w2*cost

        return cost_single_objective
    
    def cost_calculations(self, old_parameter: list, new_parameter: list):
        for individual_old, individual_new in zip(old_parameter, new_parameter):
            costInd = self.singleObject_cost(individual_old, individual_new, self.w1, self.w2)
            self.cost.append(costInd)
        return self.cost
    
class QwantaGenetic:
    def __init__(self, configuration: object):
        # Receive configuration from CreateGeneticExperiment object
        self.config = configuration

    def roulette_wheel_selection(self, population: list, fitnesses: list):
        """
        Performs roulette wheel selection on a population based on their fitnesses.
        Args:
            population (list): List of individuals.
            fitnesses (list): List of corresponding fitness values for the individuals.
            num_selected (int): Number of individuals to select.
        Returns:
            selected (list): List of selected individuals.
        """
        total_fitness = sum(fitnesses)
        probabilities = [fitness / total_fitness for fitness in fitnesses]
        
        selected = []
        while len(selected) < self.config.PARENTS_SIZE:
            # Generate a random number between 0 and 1
            r = random.uniform(0, 1)
            
            # Choose the individual whose cumulative probability includes r
            cumulative_probability = 0
            for j in range(len(population)):
                cumulative_probability += probabilities[j]
                if r <= cumulative_probability:
                    selected.append(population[j])
                    break
        
        return selected
    
    def crossover(self, parents: list):
        offspring = []
        for i in range(self.config.POPULATION_SIZE):
            #select two parents
            pool = parents
            parent1 = random.choice(pool)
            parent2 = random.choice(pool)
            #select random point to crossover
            crossover_point = random.randint(0, len(parent1)-1)
            #create child
            child = parent1[:crossover_point] + parent2[crossover_point:]
            offspring.append(child)

        return offspring
    
    def mutate(self, children: list):
        for i in range(len(children)):
            if random.random() < self.config.MUTATION_RATE:
                children[i][random.randint(0, len(children[0])-1)] = random.random()
        return children
    
    # def genetic_algorithm(self):
    #     population = self.config.population
    #     for i in range(self.config.GENERATIONS):
    #         parents = self.roulette_wheel_selection(population, self.config.cost)
    #         children = self.crossover(parents)
    #         children = self.mutate(children)
    #         population = children
    #     return population

    def genetic_algorithm(self):
        population = self.config.population

        # after end every loop, need to add Qwanta simulator
        for i in tqdm(range(self.config.GENERATIONS)):
            parents = self.roulette_wheel_selection(population, self.config.cost)
            children = self.crossover(parents)
            mutated_child = []
            for individual in children:
                mutated_child.append(self.mutate(individual))

            # Qwanta here
            experiment = QwantaSimulator(generation=i, loss=mutated_child[0], p_dep=mutated_child[1], gateErr=mutated_child[2], meaErr=mutated_child[3])

        return population

class QwantaSimulator:
    def __init__(self, generation:int, loss:list, p_dep:list, gateErr:list, meaErr:list):
        ray.init()
        self.node_info_exp = [{
            'Node 0': {'coordinate': (0, 0, 0)},
            'Node 1': {'coordinate': (100, 0, 0)},
            'Node 2': {'coordinate': (200, 0, 0)},
            'numPhysicalBuffer': 20,
            'numInternalEncodingBuffer': 20,
            'numInternalDetectingBuffer': 10,
            'numInternalInterfaceBuffer': 2,
        },]
        self.generation = generation
        self.loss = loss
        self.p_dep = p_dep
        self.gateErr = gateErr
        self.meaErr = meaErr
        self.pickle_file_name = [f'GeneticQwantaTest_exp1_gen_{self.generation}']
        self.exp_name = ['EPPS']
        

    def meaError(self, time, tau=1):
        p = (np.e**(-1*(time/tau)))/4 + 0.75
        return [p, (1- p)/3, (1- p)/3, (1- p)/3]

    @ray.remote
    def loss_vary(self, l, j):
        Quantum_topology = [{
            ('Node 0', 'Node 1'): {
            'connection-type': 'Space',
            'function': self.p_dep,
            'loss': self.loss,
            'light speed': 300000, # km/s
            'Pulse rate': 0.0001, # waiting time for next qubit (interval)
            },
            ('Node 1', 'Node 2'): {
            'connection-type': 'Space',
            'function': self.p_dep,
            'loss': self.loss,
            'light speed': 300000,
            'Pulse rate': 0.0001,
            },
            ('Node 0', 'Node 2'): {
            'connection-type': 'Space',
            'function': self.p_dep,
            'loss': self.loss,
            'light speed': 300000,
            'Pulse rate': 0.0001,
            },
        }
        for _ in self.exp_names]

        timelines = {}
        for exp_name in self.exp_names:
            e_tl, vis_a = Experiment.read_timeline_from_csv(f'experssdp.xlsx', excel=True, sheet_name=exp_name) 
            timelines[exp_name] = e_tl
            
        e_tl[2]['Resource Type'] = 'Physical'
        e_tl[2]['Edges'] = ['Node 0', 'Node 2']
        e_tl[2]['Num Trials'] = 9000

        nodes_information = {exp_name: self.nodes_info_exp[index] for index, exp_name in enumerate(self.exp_names)}
        networks = {exp_name: Quantum_topology[index] for index, exp_name in enumerate(exp_names)}
        mem_func = {exp_name: self.memError for exp_name in self.exp_names}
        gate_error = {exp_name: self.gateErr for exp_name in self.exp_names}
        measure_error = {exp_name: self.meaErr for exp_name in self.exp_names}
        sim_time = {exp_name: None for exp_name in self.exp_names}
        labels = {exp_name: 'Physical' for exp_name in self.exp_names}

        p = [0]
        exper = Experiment(networks, timelines, measurementError=measure_error, nodes_info=nodes_information, memFunc=mem_func, gateError=gate_error, simTime=sim_time,
                        parameters_set=p, collect_fidelity_history=True, repeat=1, 
                        label_records=labels,path=j, message_log='epps', progress_bar=True)

        exper.run()
        ray.shutdown()
        
def parameterTransform(loss: list, coherenceTime: list, gateErr: list, meaErr: list):
    '''
    This function change the [0-1] value into real simulation value

    # value from Bsc_Thesis
    meaErr: 0.01, 0.03, 0.05
    memoryTime: 0.25, 0.5, 1 second
    gate error 0.01, 0.03
    loss: 0.001, 0.003, 0.005, 0.007 dB/km

    # plan to normalize value in interval of previous research
    meaErr: 0 - 0.10
    memoryTime: 0 - 1
    gateError: 0 - 0.05
    loss 0.001 - 0.010

    '''
    loss = loss[0]
    coherenceTime = coherenceTime[0]
    gateErr = gateErr[0]
    meaErr = meaErr[0]

    lossSim = 0.010 - loss*0.010 # loss sim interval [0, 0.01], 0-> 0.01, 1->0
    coherenceTimeSim = coherenceTime
    gateErrSim = 0.03 - gateErr*0.03
    meaErrSim = 0.1 - meaErr*0.1
    return lossSim, coherenceTimeSim, gateErrSim, meaErrSim