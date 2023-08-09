from all_function import *
import pickle
import datetime

x = int(input('Enter configuration: '))

def main_process(
    experiment_name,
    weight1 = 0.5,
    weight2 = 0.5,
    mutationRate = 0.05, 
    crossoverRate = 0.3,
    numIndividual = 150,
    parent_size = 30,
    numGeneration = 120,
    # repeat=5,
):

    # initialize the experiment
    baseline_value = [np.random.random() for i in range(4)]
    population = create_population(population_size=numIndividual, num_parameter=4)

    fidelity_per_generation = [] # collect fidelity of current generation
    cost_value = []

    # Define an object
    experiment_result = {
        'Experiment_name': experiment_name,
        'DateCreate': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Hyperparameters': {
            'w1': weight1,
            'w2': weight2,
            'mutation_rate': mutationRate,
            'crossover_rate': crossoverRate,
            'num_population': numIndividual,
            'num_parents': parent_size,
            'num_generation': numGeneration,
        },
        'Parameter_history': {
            'loss': [],
            'gate_error': [],
            'measurement_error': [],
            'memory_time': []
        },
        'fidelity_history': [],
        'cost_history': [],
        'baseline_value': baseline_value        
        # 'simulation_repeat': repeat,
    }

    print(decorate_prompt("This is your Genetic simulation hyperparameter..."))
    print(f'Date Create: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'weight1 (Fidelity): {weight1}')
    print(f'weight2 (Cost): {weight2}')
    print(f'Mutation Rate: {mutationRate}')
    print(f'Crossover Rate: {crossoverRate}')
    print(f'Number of Individual in Population: {numIndividual}')
    print(f'Number of Parent before Crossover: {parent_size}')
    print(f'Number of Generation: {numGeneration}')
    # print(f'Number of Repeat: {repeat}')
    print(f'Baseline value: {baseline_value}')
    print()

    # loop every generation
    for g in tqdm(range(numGeneration)):
        # loop every individual in population
        # run simulation, calculate cost, and fidelity for each individual
        dum = 1
        for ind in population:
            
            print(f'Current individual: {ind}, {dum}')
            dum+=1
            loss = ind[0]
            gate_error = ind[1]
            measurement_error = ind[2]
            memory_time = ind[3]
            depo_prob = 0.03

            loss, gate_error, measurement_error, memory_time = parameterTransform(loss, gate_error, measurement_error, memory_time)

            experiment_result['Parameter_history']['loss'].append(loss)
            experiment_result['Parameter_history']['gate_error'].append(gate_error)
            experiment_result['Parameter_history']['measurement_error'].append(measurement_error)
            experiment_result['Parameter_history']['memory_time'].append(memory_time)

            # Qwanta Simulation Part
            num_hops = 2
            num_nodes = num_hops + 1

            node_info = {f'Node {i}': {'coordinate': (int(i*100), 0, 0)} for i in range(num_nodes)}
            edge_info = {
                (f'Node {i}', f'Node {i+1}'): {
                'connection-type': 'Space',
                'depolarlizing error': [1 - depo_prob, depo_prob/3, depo_prob/3, depo_prob/3],
                'loss': loss,
                'light speed': 300000,
                'Pulse rate': 0.0001,
                f'Node {i}':{
                    'gate error': gate_error,
                    'measurement error': measurement_error,
                    'memory function': memory_time
                },
                f'Node {i+1}':{
                    'gate error': gate_error,
                    'measurement error': measurement_error,
                    'memory function': memory_time
                },
                }
            for i in range(num_hops)}

            exps = Xperiment(
                timelines_path = 'network/exper_id3_selectedStats_2hops.xlsx',
                nodes_info_exp = node_info,
                edges_info_exp = edge_info,
                gate_error = gate_error,
                measurement_error = measurement_error,
                memory_time = memory_time,
                strategies_list=['0G']
            )


            # because every simualtion with the same input will give the same output. so, no need to repeat simulation.

            # collection_fidelity_for_repeat = []
            # for i in tqdm(range(repeat)):
            #     # run experiment
            #     result = exps.execute()
            #     # print(result['0G']['fidelity'])
                
            #     collection_fidelity_for_repeat.append(result['0G']['fidelity'])

            # print(f'Fidelity for this individual: {collection_fidelity_for_repeat}')
            # print(f'Avg fidelity for this individual: {np.mean(collection_fidelity_for_repeat)}')

            # test
            # if len(collection_fidelity_for_repeat) != repeat:
            #     print(collection_fidelity_for_repeat)
            #     print(repeat)
            #     raise ValueError('collection fidelity is not equal to repeat')

            # fidelity = np.mean(collection_fidelity_for_repeat)                
            # fidelity_per_generation.append(fidelity)

            result = exps.execute()
            fidelity = result['0G']['fidelity']
            fidelity_per_generation.append(fidelity)
        
            # cost calculation
            cost = (singleObject_cost(
                baseParameter=baseline_value,
                simParameter=ind,
                w1=weight1,
                w2=weight2,
                objectFidelity=0.8,
                simFidelity=fidelity
            ))
            cost_value.append(cost)

        if len(population) != numIndividual:
            print(len(population), numIndividual)
            raise ValueError('population size is not equal to numPopulation')

        # when run every individual in population
        experiment_result['fidelity_history'].append(fidelity_per_generation)
        experiment_result['cost_history'].append(cost_value)
            
        # Genetic algorithm part to create new generation

        parents = roulette_wheel_selection(population=population, costs=cost_value, parent_size=parent_size)

        crossover_sim = crossover(parents=parents, cross=crossoverRate, population_size=numIndividual)

        offspring = mutate(children=crossover_sim, mutationRate=mutationRate)

        # validate results
        if len(offspring) != len(population):
            print(f'Number of ind individuals (offspring): {len(offspring)}')
            print(f'Number of population: {len(population)}')
            raise ValueError('offspring size is not equal to population size')
        elif len(fidelity_per_generation) != len(population):
            print(f'Num fidelity: {len(fidelity_per_generation)}')
            print(f'Number of population: {len(population)}')
            raise ValueError('fidelity size is not equal to population size')
        elif len(cost_value) != len(population):
            print(f'Num cost: {len(cost_value)}')
            print(f'Number of population: {len(population)}')
            raise ValueError('cost size is not equal to population size')

        print(f'generation {g+1} finished:')
        # Show results of fidelity and cost in this generation
        print(f'Best fidelity: {np.max(fidelity_per_generation)}, Worst fidelity: {np.min(fidelity_per_generation)}, Avg fidelity: {np.mean(fidelity_per_generation)}')
        print(f'Best cost: {np.min(cost_value)}, Worst cost: {np.max(cost_value)}, Avg cost: {np.mean(cost_value)}')

        # clear dummy list
        cost_value = []
        fidelity_per_generation = []

        population = offspring # next generation

    with open(f"results/{experiment_name}", "wb") as f:
        pickle.dump(experiment_result, f)

    print('Simulation finished')

# input_params = [
#     {'experiment_name': 'exp_HyperTune1.pickle', 'weight1': 0.2, 'weight2': 0.8, 'mutationRate': 0.05, 'crossoverRate': 0.3},
#     {'experiment_name': 'exp_HyperTune2.pickle', 'weight1': 0.5, 'weight2': 0.5, 'mutationRate': 0.05, 'crossoverRate': 0.3},
#     {'experiment_name': 'exp_HyperTune3.pickle', 'weight1': 0.8, 'weight2': 0.2, 'mutationRate': 0.05, 'crossoverRate': 0.3},
#     {'experiment_name': 'exp_HyperTune4.pickle', 'weight1': 0.5, 'weight2': 0.5, 'mutationRate': 0.10, 'crossoverRate': 0.3},
#     {'experiment_name': 'exp_HyperTune5.pickle', 'weight1': 0.5, 'weight2': 0.5, 'mutationRate': 0.15, 'crossoverRate': 0.3},
#     {'experiment_name': 'exp_HyperTune6.pickle', 'weight1': 0.5, 'weight2': 0.5, 'mutationRate': 0.05, 'crossoverRate': 0.1},
#     {'experiment_name': 'exp_HyperTune7.pickle', 'weight1': 0.5, 'weight2': 0.5, 'mutationRate': 0.05, 'crossoverRate': 0.5},
#     {'experiment_name': 'fast_test.pickle', 'weight1': 0.5, 'weight2': 0.5, 'mutationRate': 0.05, 'crossoverRate': 0.3},
# ]

input_params = [    
    {'experiment_name': 'exp_mr_5', 'weight1': 0.5, 'weight2': 0.5, 'mutationRate': 0.05, 'crossoverRate': 0.3},
    {'experiment_name': 'exp_mr_10', 'weight1': 0.5, 'weight2': 0.5, 'mutationRate': 0.10, 'crossoverRate': 0.3},
    {'experiment_name': 'exp_mr_15', 'weight1': 0.5, 'weight2': 0.5, 'mutationRate': 0.15, 'crossoverRate': 0.3},
    {'experiment_name': 'exp_mr_20', 'weight1': 0.5, 'weight2': 0.5, 'mutationRate': 0.20, 'crossoverRate': 0.3},
    {'experiment_name': 'exp_mr_25', 'weight1': 0.5, 'weight2': 0.5, 'mutationRate': 0.25, 'crossoverRate': 0.3},
    {'experiment_name': 'exp_mr_30', 'weight1': 0.5, 'weight2': 0.5, 'mutationRate': 0.30, 'crossoverRate': 0.3},
    {'experiment_name': 'exp_mr_35', 'weight1': 0.5, 'weight2': 0.5, 'mutationRate': 0.35, 'crossoverRate': 0.3},
    {'experiment_name': 'exp_mr_40', 'weight1': 0.5, 'weight2': 0.5, 'mutationRate': 0.40, 'crossoverRate': 0.3},
    
    {'experiment_name': 'weight_01_09', 'weight1': 0.1, 'weight2': 0.9, 'mutationRate': 0.05, 'crossoverRate': 0.3},
    {'experiment_name': 'weight_02_08', 'weight1': 0.2, 'weight2': 0.8, 'mutationRate': 0.05, 'crossoverRate': 0.3},
    {'experiment_name': 'weight_03_07', 'weight1': 0.3, 'weight2': 0.7, 'mutationRate': 0.05, 'crossoverRate': 0.3},
    {'experiment_name': 'weight_04_06', 'weight1': 0.4, 'weight2': 0.6, 'mutationRate': 0.05, 'crossoverRate': 0.3},
    {'experiment_name': 'weight_06_04', 'weight1': 0.6, 'weight2': 0.4, 'mutationRate': 0.05, 'crossoverRate': 0.3},
    {'experiment_name': 'weight_07_03', 'weight1': 0.7, 'weight2': 0.3, 'mutationRate': 0.05, 'crossoverRate': 0.3},
    {'experiment_name': 'weight_08_02', 'weight1': 0.8, 'weight2': 0.2, 'mutationRate': 0.05, 'crossoverRate': 0.3},
    {'experiment_name': 'weight_09_01', 'weight1': 0.9, 'weight2': 0.1, 'mutationRate': 0.05, 'crossoverRate': 0.3},
]  

if __name__ == '__main__':
    params = input_params[x]
    main_process(**params)
