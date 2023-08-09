from all_function import *
import pickle

x = int(input('Enter configuration: '))

def main_process(
    experiment_name,
    weight1 = 0.5,
    weight2 = 0.5,
    # mutationRate = 0.05,
    # crossoverRate = 0.6,
    numPopulation = 150,
    # parent_size = 30,
    numGeneration = 50
):

    # initialize the experiment
    baseline_value = [np.random.random() for i in range(4)]
    population = create_population(population_size=numPopulation, num_parameter=4)

    fidelity_per_generation = [] # collect fidelity of current generation
    cost_value = []

    # Define an object
    experiment_result = {
        'Experiment_name': experiment_name,
        'Hyperparameters': {
            'w1': weight1,
            'w2': weight2,
            # 'mutation_rate': mutationRate,
            # 'crossover_rate': crossoverRate,
            'num_population': numPopulation,
            # 'num_parents': parent_size,
            'num_generation': numGeneration,
        },
        'fidelity_history': [],
        'cost_history': [],
    }

    print(decorate_prompt("This is your Genetic simulation hyperparameter..."))
    print(f'weight1: {weight1}')
    print(f'weight2: {weight2}')
    # print(f'mutationRate: {mutationRate}')
    # print(f'crossoverRate: {crossoverRate}')
    print(f'numPopulation: {numPopulation}')
    # print(f'parent_size: {parent_size}')
    print(f'numGeneration: {numGeneration}')
    print()

    for g in tqdm(range(numGeneration)):
    # loop every individual in population
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

            # run experiment
            result = exps.execute()

            # fidelity
            fidelity = result['0G']['fidelity']
            fidelity_per_generation.append(fidelity)
        
            # cost
            cost = (singleObject_cost(
                baseParameter=baseline_value,
                simParameter=ind,
                w1=weight1,
                w2=weight2,
                objectFidelity=0.8,
                simFidelity=fidelity
            ))
            cost_value.append(cost)
            # print(f'cost value: {cost_value}')

        experiment_result['fidelity_history'].append(fidelity_per_generation)
        experiment_result['cost_history'].append(cost_value)
            
        
        # parents = roulette_wheel_selection(population=population, costs=cost_value, parent_size=parent_size)
        selected, selected_mutate = roulette_wheel_selection_SS(population=population, costs=cost_value, numSelected=100, numMutated=30)

        # crossover_sim = crossover(parents=parents, cross=crossoverRate, population_size=numPopulation)
        offspring = crossover_SS(population_size=numPopulation, selected=selected, selected_mutate=selected_mutate)

        # offspring = mutate(children=crossover_sim, mutationRate=mutationRate)
        # print(f'offspring: {offspring}')

        # clear dummy list
        cost_value = []
        fidelity_per_generation = []

        population = offspring # 2nd generation

        print(f'generation {g+1} finished:')
        print(f'Best fidelity: {np.max(fidelity)}, Worst fidelity: {np.min(fidelity)}, Avg fidelity: {np.mean(fidelity)}')
        print(f'Best cost: {np.min(cost)}, Worst cost: {np.max(cost)}, Avg cost: {np.mean(cost)}')

    with open(f"results/{experiment_name}", "wb") as f:
        pickle.dump(experiment_result, f)

    print('Simulation finished')

input_params = [
    {'experiment_name': 'exp_HyperTune.pickle', 'weight1': 0.3, 'weight2': 0.7},
    {'experiment_name': 'exp_HyperTune.pickle', 'weight1': 0.5, 'weight2': 0.5},
    {'experiment_name': 'exp_HyperTune.pickle', 'weight1': 0.7, 'weight2': 0.3},

    {'experiment_name': 'exp_HyperTune.pickle', 'weight1': 0.5, 'weight2': 0.5},
    {'experiment_name': 'exp_HyperTune.pickle', 'weight1': 0.5, 'weight2': 0.5},
    {'experiment_name': 'exp_HyperTune.pickle', 'weight1': 0.5, 'weight2': 0.5},
    {'experiment_name': 'exp_HyperTune.pickle', 'weight1': 0.5, 'weight2': 0.5},
]  

# for params in input_params:
#         main_process(**params)

if __name__ == '__main__':
    params = input_params[x]
    main_process(**params)

# if __name__ == '__main__':
#     main_process()