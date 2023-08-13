import sys
sys.path.insert(0, '/Users/poramat/Documents/qwanta/tutorial/poramat_work_part/Genetic_algorithm_development')
sys.path.insert(1, '/Users/poramat/Documents/qwanta/tutorial/poramat_work_part')
sys.path.insert(2, '/Users/poramat/Documents/qwanta')
sys.path.insert(3, '/Users/poramat/Documents/qwanta/tutorial/network')
from ga_Develop_I import *
from all_function import *
from qwanta import Xperiment
from tqdm import tqdm

def main_process(
    experiment_name='Experiment_1',
    elitism = 0.2,
    population_size = 500,
    mutation_rate = 0.8,
    mutation_sigma = 0.3,
    mutation_decay = 0.999,
    mutation_limit = 0.01,
    amount_optimisation_steps = 400,
    dna_bounds = (0, 1),
    dna_start_position = [0, 0, 0, 0],
    weight1 = 0.5,
    weight2 = 0.5,
    objective_fidelity = 0.7
):    

    ga = GA_Develop_I(
        dna_size=len(dna_start_position),
        dna_bounds=dna_bounds,
        dna_start_position=dna_start_position,
        elitism=elitism,
        population_size=population_size,
        mutation_rate=mutation_rate,
        mutation_sigma=mutation_sigma,
        mutation_decay=mutation_decay,
        mutation_limit=mutation_limit,
    )

    experiment_result = ExperimentResult(
        experiment_name=experiment_name, 
        weight1=weight1, 
        weight2=weight2, 
        mutation_rate=mutation_rate,
        population_size=mutation_sigma, 
        elitism=elitism, 
        amount_optimization_steps=amount_optimisation_steps,                       
    )

    decorated_prompt = decorate_prompt(
        prompt = "This is your Genetic simulation hyperparameter...",
        weight1 = weight1,
        weight2 = weight2,
        mutationRate = mutation_rate,
        numIndividual = population_size,
        parent_size = int(population_size*elitism),
        numGeneration = amount_optimisation_steps,        
    )
    print(decorated_prompt)

    # In this version, I define two way for baseline value
    # 1. random value in every single individual
    baseline_value = [[np.random.random() for i in range(4)] for j in range(ga.population_size)]
    # 2. random only one ind and use the same for every ind (this ways is the same for mainG.py)
    # baseline_value = [np.random.random() for i in range(4)]

    # photon loss (db/km): 
    photon_loss = [ga.population[i][0] for i in range(ga.population_size)]
    # gate error: (%)
    gate_error = [ga.population[i][1] for i in range(ga.population_size)]
    # measurement error: (%)
    measurement_error = [ga.population[i][2] for i in range(ga.population_size)]
    # coherence time (seconds):
    coherence = [ga.population[i][3] for i in range(ga.population_size)]
        
    optimize_data = [[
        photon_loss[i], 
        coherence[i], 
        gate_error[i], 
        measurement_error[i]
    ] for i in range(ga.population_size)]

    for step in tqdm(range(amount_optimisation_steps), desc='Optimizing...'):
        
        fidelity_array = []
        cost_array = []
        index = 0
        for loss_parameter in tqdm(optimize_data, desc='Simulating...'):

            assert len(loss_parameter) == 4
            
            # In this part, loss must be [0, 1] value
            loss = loss_parameter[0]
            memory_time = loss_parameter[1]
            gate_error = loss_parameter[2]
            measurement_error = loss_parameter[3]            
            depo_prob = 0.03            

            # Transform parameter
            Tloss, Tmemory_time, Tgate_error, Tmeasurement_error = parameterTransform(
                loss, memory_time, gate_error, measurement_error
            )

            print(f'Loss {loss, memory_time, gate_error, measurement_error}')
            print(f'Transform loss {Tloss, Tmemory_time, Tgate_error, Tmeasurement_error}')

            # Qwanta Simulation Part
            num_hops = 2
            num_nodes = num_hops + 1

            node_info = {f'Node {i}': {'coordinate': (int(i*100), 0, 0)} for i in range(num_nodes)}
            edge_info = {
                (f'Node {i}', f'Node {i+1}'): {
                'connection-type': 'Space',
                'depolarlizing error': [1 - depo_prob, depo_prob/3, depo_prob/3, depo_prob/3],
                'loss': Tloss,
                'light speed': 300000,
                'Pulse rate': 0.0001,
                f'Node {i}':{
                    'gate error': Tgate_error,
                    'measurement error': Tmeasurement_error,
                    'memory function': Tmemory_time
                },
                f'Node {i+1}':{
                    'gate error': Tgate_error,
                    'measurement error': Tmeasurement_error,
                    'memory function': Tmemory_time
                },
                }
            for i in range(num_hops)}

            exps = Xperiment(
                timelines_path = '../network/exper_id3_selectedStats_2hops.xlsx',
                nodes_info_exp = node_info,
                edges_info_exp = edge_info,
                gate_error = Tgate_error,
                measurement_error = Tmeasurement_error,
                memory_time = Tmemory_time,
                strategies_list=['0G']
            )

            result = exps.execute()
            fidelity = result['0G']['fidelity']            

            # Check parameter carefully in every loop
            cost = singleObject_cost(
                baseParameter=baseline_value[index],
                simParameter=loss_parameter,
                w1=weight1,
                w2=weight2,
                objectFidelity=objective_fidelity,
                simFidelity=fidelity
            )

            # Fidelity array will correct all fidelity in one Generation
            fidelity_array.append(fidelity)
            cost_array.append(cost)
            index += 1                        

        assert len(fidelity_array) == len(optimize_data) == len(cost_array) == ga.population_size

        experiment_result['Fidelity_history'].append(fidelity_array)

        # Genetic Algorithm Part
        best_dna, lowest_cost = ga.evolve(np.array(cost_array))

        # New population
        loss_new = [ga.population[i][0] for i in range(ga.population_size)]
        memory_time_new = [ga.population[i][1] for i in range(ga.population_size)]
        gate_error_new = [ga.population[i][2] for i in range(ga.population_size)]
        measurement_error_new = [ga.population[i][3] for i in range(ga.population_size)]

        # update simulation parameter
        optimize_data = [loss_new, memory_time_new, gate_error_new, measurement_error_new]
        baseline_value = ga.population

        assert np.array(optimize_data).shape == (4, ga.population_size)

        print(f'Generation {step+1} | Lowest cost: {lowest_cost} | Best DNA: {best_dna}')

    # Add ga object and Save result
    experiment_result.add_ga_object(ga)

    path = f'results/{experiment_name}'
    experiment_result.save(file_name=path)

if __name__ == '__main__':
    config_filename = "config.json"
    config = read_config(config_filename)

    experiment_name = config["experiment_name"]
    elitism = config["elitism"]
    population_size = config["population_size"]
    mutation_rate = config["mutation_rate"]
    mutation_sigma = config["mutation_sigma"]
    mutation_decay = config["mutation_decay"]
    mutation_limit = config["mutation_limit"]
    amount_optimisation_steps = config["amount_optimisation_steps"]
    dna_bounds = config["dna_bounds"]
    dna_start_position = config["dna_start_position"]
    weight1 = config["weight1"]
    weight2 = config["weight2"]
    objective_fidelity = config["objective_fidelity"]

    main_process(
        experiment_name,
        elitism,
        population_size,
        mutation_rate,
        mutation_sigma,
        mutation_decay,
        mutation_limit,
        amount_optimisation_steps,
        dna_bounds,
        dna_start_position,
        weight1,
        weight2,
        objective_fidelity
    )
