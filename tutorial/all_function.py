from qwanta import Tuner, QuantumNetwork, Configuration, Xperiment
from qwanta import QuantumNetwork, Configuration, Xperiment, Tuner
import pandas as pd
import ast
import numpy as np
import networkx as nx
from qwanta.Qubit import PhysicalQubit
from tqdm import tqdm
import random
import csv
import pickle
import matplotlib.pyplot as plt

def multinomial_argmax(probabilities):
    # normalize probabilities to sum to 1
    probabilities = np.asarray(probabilities)
    probabilities /= probabilities.sum()

    # draw a sample from the multinomial distribution
    sample = np.random.multinomial(1, probabilities)

    # get the index of the selected element
    index = np.argmax(sample)

    return index

def create_population(population_size, num_parameter=4):
    # create individual in population
    population = []
    for i in range(population_size):
        individual = [random.random() for i in range(num_parameter)]
        population.append(individual)
    
    return population
    
def Heaviside(x):
    if x >= 0:
        return 1
    else:
        return 0
    
# This calculate cost for one individual
def singleObject_cost(baseParameter, simParameter, w1, w2, objectFidelity, simFidelity):
    # cost function
    output = []
    for i in range(len(baseParameter)):
        k = np.log(baseParameter[i])/np.log(simParameter[i])
        output.append(k)

    cost = sum(output)

    cost_single_objective = w1*Heaviside(objectFidelity - simFidelity) + w2*cost

    return cost_single_objective

def visualize(result):
    to_print = ['fidelity', 'simulation_time', 'Resources Produced', 'Base Resources Attempt', 'Resource Used in Fidelity Estimation', 'Time used', 'Fidelity Estimation Time']
    for key in to_print:
        print(f'{key}: {result[key]}')

def normalize_list(l):
    return [i/sum(l) for i in l]

def roulette_wheel_selection(population, costs, parent_size):
    """
        Performs roulette wheel selection on a population based on their fitnesses.
        Args:
            population (list): List of individuals.
            fitnesses (list): List of corresponding fitness values for the individuals.
            num_selected (int): Number of individuals to select.
        Returns:
            selected (list): List of selected individuals.
    """

    # prevent changing original population
    dum_population = population[:]

    fitness_list = [1 / (1 + cost) for cost in costs]

    probabilities = [fitness/sum(fitness_list) for fitness in fitness_list]
    # print(f'probabilities: {probabilities}, and sum is: {sum(probabilities)}')

    # checked that sum(probabilities) = 1
    if sum(probabilities) < 0.999:
        print(f'Sum Probability: {sum(probabilities)}')
        raise ValueError('Probabilities must sum to 1.')
    elif sum(probabilities) > 1.001:
        print(f'Sum Probability: {sum(probabilities)}')
        raise ValueError('Probabilities must not over 1.')
        
    selected = []
    
    # print('start selecting parents')
    while len(selected) < parent_size:
    
        index = multinomial_argmax(probabilities)        
        chosen_parent = dum_population[index]

        selected.append(chosen_parent)

        del dum_population[index]
        del probabilities[index]
        
    return selected

def roulette_wheel_selection_SS(population, costs,  numSelected, numMutated, mutateRateHigh=0.1, mutateRateLow=0.05):
    fitness_list = [1 / (1 + cost) for cost in costs]
    probabilities = [fitness/sum(fitness_list) for fitness in fitness_list]
    prob_to_mutate = []

     # checked that sum(probabilities) = 1
    if sum(probabilities) < 0.999:
        print(f'Sum Probability: {sum(probabilities)}')
        raise ValueError('Probabilities must sum to 1.')
    elif sum(probabilities) > 1.001:
        print(f'Sum Probability: {sum(probabilities)}')
        raise ValueError('Probabilities must not over 1.')
    
    selected = []
    selected_mutate = []
    mutate_offspring = []

    while len(selected) < numSelected:
        index = multinomial_argmax(probabilities)
        chosen_parent = population[index]

        selected.append(chosen_parent)

        del population[index]
        del probabilities[index]

    if len(selected) != numSelected:
        print(f'len selected: {len(selected)}')
        raise ValueError('Selected parents must be 100')

    while len(selected_mutate) < numMutated:
        index = multinomial_argmax(probabilities)
        
        selected_mutate.append(population[index])
        prob_to_mutate.append(probabilities[index])

        del population[index]
        del probabilities[index]

    if len(selected_mutate) != numMutated:
        print(f'len selected_mutate: {len(selected_mutate)}')
        raise ValueError('Selected parents must be 30')

    numInd_high_mutate = int(numMutated/2)
    numInd_low_mutate = numMutated - numInd_high_mutate

    # fill low mutate (high strength) individuals into mutate offspring
    for i in range(numInd_low_mutate):
        # select index of high prob in selected_mutate. the selected will have less mutate rate (high strength, low mutate)
        index = multinomial_argmax(prob_to_mutate)
        mutate_offspring.append(selected_mutate[index])

        del selected_mutate[index]
        del prob_to_mutate[index]
        # when this loop is finished, selected_mutate will left only low mutate individuals
        # high mutate => mutate_offspring
        # low mutate => selected_mutate

    if len(mutate_offspring) + len(selected_mutate) != numMutated:
        raise ValueError('Something wrong with selection 1')


    # mutate low
    for i in range(numInd_low_mutate):
        for j in range(4):
            if random.random() < mutateRateLow:
                selected_mutate[i][j] = random.random() # because now mutate offspring contain only low mutate individuals

    if len(selected_mutate) != numInd_low_mutate:
        print(f'len(selected_mutate): {len(selected_mutate)}')
        raise ValueError('Something wrong with mutate low')

    # mutate high
    for i in range(numInd_high_mutate):
        for j in range(4):
            if random.random() < mutateRateHigh:
                mutate_offspring[i][j] = random.random()
    
    if len(mutate_offspring) != numInd_high_mutate:
        print(f'len(mutate_offspring): {len(mutate_offspring)}')
        raise ValueError('Something wrong with mutate high')

    if len(selected) + len(selected_mutate) + len(mutate_offspring) != numSelected + numMutated:
        print(f'len(selected): {len(selected)}')
        print(f'len(selected_mutate): {len(selected_mutate)}')
        print(f'len(mutate_offspring): {len(mutate_offspring)}')
        raise ValueError('Something wrong with selection 2')

    if len(selected_mutate) + len(mutate_offspring) != numMutated:
        print(f'len(selected_mutate): {len(selected_mutate)}')
        print(f'len(mutate_offspring): {len(mutate_offspring)}')
        print(f'numMutated: {numMutated}')
        raise ValueError('Something wrong with mutate offspring')
            
    # after mutate, combind mutate_offspring and selected_mutate
    selected_mutate = selected_mutate + mutate_offspring

    return selected, selected_mutate

def crossover_SS(population_size, selected, selected_mutate, numSelected, numMutated):
    offspring = []

    # add select(100) into offspring
    offspring.append(selected)

    if offspring != numSelected:
        print(f'len(offspring): {len(offspring)}')
        print(f'numSelected: {numSelected}')
        raise ValueError('Check here 1')
    
    if len(selected_mutate) != numMutated:
        print(f'len(selected_mutate): {len(selected_mutate)}')
        print(f'numMutated: {numMutated}')
        raise ValueError('selected_mutate not equal numMutated')
    
    num_cross = population_size - len(selected) - len(selected_mutate)

    if num_cross != population_size - numSelected - numMutated:
        raise ValueError('Check here 2')
    
    # type: crossover
    for i in range(num_cross):
        pool = selected
        parent1 = random.choice(pool)
        parent2 = random.choice(pool)

        # select random point to crossover
        crossover_point = random.randint(0, len(parent1)-1)
            
        # create child
        child = parent1[:crossover_point] + parent2[crossover_point:]
        # add crossover(20) into offspring
        offspring.append(child)

    # must be 120 = 150-30
    if len(offspring) != population_size - numMutated:
        print(f'len(offspring): {len(offspring)}')
        print(f'population_size: {population_size}')
        print(f'numMutated: {numMutated}')
        raise ValueError('Check here 3')

    # add mutate(30) into offspring
    offspring = offspring + selected_mutate

    if len(offspring) != population_size:
        print(f'len(offspring): {len(offspring)}')
        print(f'population_size: {population_size}')
        raise ValueError('Something wrong with offspring')

    return offspring

def crossover(parents, cross, population_size):
    offspring = []
    numParent = len(parents)

    num_cross = int(cross*population_size)
    num_pick = population_size - num_cross

    for i in range(num_cross):
        
        #select two parents
        pool = parents
        parent1 = random.choice(pool)
        parent2 = random.choice(pool)

        #select random point to crossover
        crossover_point = random.randint(0, len(parent1)-1)        
            
        #create child
        child = parent1[:crossover_point] + parent2[crossover_point:]
        offspring.append(child)

    for i in range(num_pick):
        picked_ind = random.choice(parents)
        offspring.append(picked_ind)
        # del parents[parents.index(picked_ind)]

    if num_cross+num_pick != population_size:
        print(f'Num cross: {num_cross}, Num pick: {num_pick}, Num parents: {numParent}')
        raise ValueError('Number is not match in crossover function parent')
    
    elif num_cross+num_pick != len(offspring):
        print(f'Num cross: {num_cross}, Num pick: {num_pick}, Num offspring: {len(offspring)}')
        raise ValueError('Number is not match in crossover function offspring')

    return offspring

def mutate(children, mutationRate=0.05):
    for i in range(len(children)):
        
        if random.random() < mutationRate:
            children[i][random.randint(0, len(children[0])-1)] = random.random()
    return children

def parameterTransform(loss, coherenceTime, gateErr, meaErr):
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
    # loss = loss[0]
    # coherenceTime = coherenceTime[0]
    # gateErr = gateErr[0]
    # meaErr = meaErr[0]

    lossSim = 0.010 - loss*0.010 # loss sim interval [0, 0.01], 0-> 0.01, 1->0
    coherenceTimeSim = coherenceTime
    gateErrSim = 0.03 - gateErr*0.03
    meaErrSim = 0.1 - meaErr*0.1
    return lossSim, coherenceTimeSim, gateErrSim, meaErrSim

def decorate_prompt(prompt):
    """
    A helper function to decorate the prompt with asterisks
    """
    width = len(prompt) + 4
    return f"{'*' * width}\n  {prompt}\n{'*' * width}\n"

def read_pickle(fileName, plotFC=False, plotF=False, plotC=False):
    with open(fileName, "rb") as f:
        my_loaded_object = pickle.load(f)
    
    x = np.arange(0, my_loaded_object['Hyperparameters']['num_generation'], 1)
    w1 = my_loaded_object['Hyperparameters']['w1']
    w2 = my_loaded_object['Hyperparameters']['w2']
    mutation_rate = my_loaded_object['Hyperparameters']['mutation_rate']
    crossover_rate = my_loaded_object['Hyperparameters']['crossover_rate']
    num_population = my_loaded_object['Hyperparameters']['num_population']
    num_parents = my_loaded_object['Hyperparameters']['num_parents']
    num_generation = my_loaded_object['Hyperparameters']['num_generation']
    # fidelity = my_loaded_object['fidelity_history']
    # cost = my_loaded_object['cost_history']

    mean_fidelity = [np.mean(fidelity) for fidelity in my_loaded_object['fidelity_history']]
    mean_cost = [np.mean(cost) for cost in my_loaded_object['cost_history']]

    round_fidelity = []
    for fidelity in my_loaded_object['fidelity_history'][0]:
        new_fidelity = round(fidelity, 3)
        round_fidelity.append(new_fidelity)
        
    mode_fidelity = [max(set(round_fidelity), key=round_fidelity.count) for round_fidelity in my_loaded_object['fidelity_history']]

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    if plotFC == True:                
        ax[0, 0].set_title(f'w1: {w1}, w2: {w2}, mr: {mutation_rate}, cr: {crossover_rate}, pop: {num_population}, parents: {num_parents}, ng: {num_generation}', loc='right')

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

        ax[0, 1].plot(my_loaded_object['fidelity_history'][0], my_loaded_object['cost_history'][0], '.')
        ax[0, 1].set_xlabel('fidelity')
        ax[0, 1].set_ylabel('cost')

    if plotF == True:
        ax[1, 0].set_title('Fidelity of each generation')
        ax[1, 0].set_xlabel('generation')
        ax[1, 0].set_ylabel('fidelity')
        ax[1, 0].plot(x, np.max(my_loaded_object['fidelity_history'], axis=1), label='max fidelity', color='tab:red')
        ax[1, 0].plot(x, np.min(my_loaded_object['fidelity_history'], axis=1), label='min fidelity', color='tab:green')
        ax[1, 0].plot(x, np.mean(my_loaded_object['fidelity_history'], axis=1), label='mean fidelity', color='tab:blue')
        ax[1, 0].plot(x, mode_fidelity, label='mode fidelity', color='tab:orange')
        ax[1, 0].fill_between(x, np.max(my_loaded_object['fidelity_history'], axis=1), np.min(my_loaded_object['fidelity_history'], axis=1), alpha=0.2, label='range fidelity')
        ax[1, 0].legend()
        

    if plotC == True:
        # ax[1, 1].set_title('Cost of each generation')
        # ax[1, 1].set_xlabel('generation')
        # ax[1, 1].set_ylabel('cost')
        # # ax[1, 1].plot(x, np.max(my_loaded_object['cost_history'], axis=1), label='max cost', color='tab:red')
        # ax[1, 1].plot(x, np.min(my_loaded_object['cost_history'], axis=1), label='min cost', color='tab:green')
        # ax[1, 1].plot(x, np.mean(my_loaded_object['cost_history'], axis=1), label='mean cost', color='tab:blue')
        # # ax[1, 1].fill_between(x, np.max(my_loaded_object['cost_history'], axis=1), np.min(my_loaded_object['cost_history'], axis=1), alpha=0.2, label='range cost')
        # ax[1, 1].legend()

        ax[1, 1].set_title('Evolution of cost')

        color = 'tab:red'
        ax[1, 1].set_xlabel('generation')
        ax[1, 1].set_ylabel('avg cost', color=color)
        ax[1, 1].plot(x, np.mean(my_loaded_object['cost_history'], axis=1), 'r.-', label='mean cost', color=color)
        ax[1, 1].tick_params(axis='y', labelcolor=color)

        ax2 = ax[1, 1].twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('best cost', color=color)  # we already handled the x-label with ax1
        ax[1, 1].plot(x, np.min(my_loaded_object['cost_history'], axis=1), 'b+',label='min cost', color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        ax[1, 1].legend()
        
        

    # if plotDot == True:
        
    #     plt.plot(my_loaded_object['fidelity_history'][0], my_loaded_object['cost_history'][0], '.')
    #     plt.xlabel('fidelity')
    #     plt.ylabel('cost')
        