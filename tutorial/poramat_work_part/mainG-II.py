from Genetic_algorithm_development.ga_Develop_I import *
from Genetic_algorithm_development.all_function import *
from qwanta import Xperiment
from tqdm import tqdm
import os
import multiprocessing

def main_process(
    experiment_name,
    elitism = 0.2,
    population_size = 500,
    mutation_rate = 0.8,
    mutation_sigma = 0.3,
    mutation_decay = 0.999,
    mutation_limit = 0.01,
    amount_optimization_steps = 400,
    dna_bounds = (0, 1),
    dna_start_position = [0, 0, 0, 0],
    weight1 = 0.5,
    weight2 = 0.5,
    objective_fidelity = 0.7,
    num_hops = 2,
    excel_file = "exper_id3_selectedStats_2hops.xlsx",
    strategy = "0G"
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
        strategy=strategy,
        weight1=weight1, 
        weight2=weight2, 
        mutation_rate=mutation_rate,
        population_size=mutation_sigma, 
        elitism=elitism, 
        amount_optimization_steps=amount_optimization_steps,                       
    )

    decorated_prompt = decorate_prompt(
        prompt = "This is your Genetic simulation hyperparameter...",
        experiment_name = experiment_name,
        weight1 = weight1,
        weight2 = weight2,
        mutationRate = mutation_rate,
        numIndividual = population_size,
        parent_size = int(population_size*elitism),
        numGeneration = amount_optimization_steps,  
        strategy = strategy,      
        num_hops = num_hops,
    )
    print(decorated_prompt)

    # In this version, I define two way for baseline value
    # 1. random value in every single individual
    baseline_value = [[np.random.random() for i in range(4)] for j in range(ga.population_size)]
    # 2. random only one ind and use the same for every ind (this ways is the same for mainG.py)
    # baseline_value = [np.random.random() for i in range(4)]

    # photon loss (db/km): 
    photon_loss = [ga.population[i][0] for i in range(ga.population_size)]
    # coherence time (seconds):
    coherence = [ga.population[i][1] for i in range(ga.population_size)]
    # gate error: (%)
    gate_error = [ga.population[i][2] for i in range(ga.population_size)]
    # measurement error: (%)
    measurement_error = [ga.population[i][3] for i in range(ga.population_size)]
    
        
    optimize_data = [[
        photon_loss[i], 
        coherence[i], 
        gate_error[i], 
        measurement_error[i]
    ] for i in range(ga.population_size)]

    # Number of generation
    for step in tqdm(range(amount_optimization_steps), desc='Optimizing step... '):
        
        fidelity_array = []
        cost_array = []
        index = 0

        # Number of individual in population
        for loss_parameter in optimize_data:
            
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
                timelines_path = f'../network/{excel_file}',
                nodes_info_exp = node_info,
                edges_info_exp = edge_info,
                gate_error = Tgate_error,
                measurement_error = Tmeasurement_error,
                memory_time = Tmemory_time,
                strategies_list=[strategy]
            )

            result = exps.execute()

            # This is fidelity of one individual            
            fidelity = result[strategy]['fidelity']                     

            # Check parameter carefully in every loop
            cost = singleObject_cost(
                baseParameter=baseline_value[index],
                simParameter=loss_parameter,
                w1=weight1,
                w2=weight2,
                objectFidelity=objective_fidelity,
                simFidelity=fidelity
            )            

            # Fidelity array will collect all fidelity in one Generation
            fidelity_array.append(fidelity)
            cost_array.append(cost)
            index += 1
            
        assert len(fidelity_array) == len(optimize_data) == len(cost_array) == ga.population_size

        max_fidelity_generation = np.max(fidelity_array)
        mean_fidelity_generation = np.mean(fidelity_array)
        min_fidelity_generation = np.min(fidelity_array)
        max_cost_generation = np.max(cost_array)
        mean_cost_generation = np.mean(cost_array)        
        min_cost_generation = np.min(cost_array)

        index_max_fidelity = np.argmax(fidelity_array)
        best_parameter_generation = optimize_data[index_max_fidelity]

        experiment_result.experiment_config['Parameter_history']['loss'].append(best_parameter_generation[0])
        experiment_result.experiment_config['Parameter_history']['memory_time'].append(best_parameter_generation[1])
        experiment_result.experiment_config['Parameter_history']['gate_error'].append(best_parameter_generation[2])
        experiment_result.experiment_config['Parameter_history']['measurement_error'].append(best_parameter_generation[3])

        experiment_result.experiment_config['fidelity_history']['max'].append(max_fidelity_generation)
        experiment_result.experiment_config['fidelity_history']['mean'].append(mean_fidelity_generation)
        experiment_result.experiment_config['fidelity_history']['min'].append(min_fidelity_generation)
        experiment_result.experiment_config['cost_history']['max'].append(max_cost_generation)
        experiment_result.experiment_config['cost_history']['mean'].append(mean_cost_generation)
        experiment_result.experiment_config['cost_history']['min'].append(min_cost_generation)        

        # Genetic Algorithm Part        

        # This will be next baseline value (population before evole)
        baseline_value = ga.population
        print(f'Baseline value {baseline_value}')

        best_dna, lowest_cost = ga.evolve(np.array(cost_array))

        # This will be next sim value (population after evole)
        optimize_data = [
            [ga.population[i][0], ga.population[i][1], ga.population[i][2], ga.population[i][3]] for i in range(ga.population_size)
        ]
        print(f'Optimize data {optimize_data}')        

        assert np.array(optimize_data).shape == (ga.population_size, 4)

        print(f'Generation {step+1} | Lowest cost: {lowest_cost} | Best DNA: {best_dna}')

    # Add ga object and Save result
    experiment_result.add_ga_object(ga)
    
    path = f'results'
    experiment_result.save(file_path=path, file_name=experiment_name)

def process_config(config_filename):
    config = read_config("configs/" + config_filename)

    experiment_name = config["experiment_name"]
    elitism = config["elitism"]
    population_size = config["population_size"]
    mutation_rate = config["mutation_rate"]
    mutation_sigma = config["mutation_sigma"]
    mutation_decay = config["mutation_decay"]
    mutation_limit = config["mutation_limit"]
    amount_optimization_steps = config["amount_optimization_steps"]
    dna_bounds = config["dna_bounds"]
    dna_start_position = config["dna_start_position"]
    weight1 = config["weight1"]
    weight2 = config["weight2"]
    objective_fidelity = config["objective_fidelity"]
    num_hops = config["num_hops"]
    excel_file = config["excel_file"]
    strategy = config["strategy"]

    main_process(
        experiment_name,
        elitism,
        population_size,
        mutation_rate,
        mutation_sigma,
        mutation_decay,
        mutation_limit,
        amount_optimization_steps,
        dna_bounds,
        dna_start_position,
        weight1,
        weight2,
        objective_fidelity,
        num_hops,
        excel_file,
        strategy
    )

if __name__ == '__main__':
    config_directory = "configs/"  # Update this to the directory where your config files are located
    
    # List all the files in the config directory
    config_files = os.listdir(config_directory)
    
    # Ask the user for their choice
    user_choice = input("Run all config file? (Y/n)").strip().lower()
    
    if user_choice == 'y':
        # Create a multiprocessing pool with the number of desired parallel processes
        num_processes = multiprocessing.cpu_count()  # Use all available CPU cores
        pool = multiprocessing.Pool(processes=num_processes)
        
        # Use pool.map to execute process_config in parallel for each config file
        pool.map(process_config, [filename for filename in config_files if filename.endswith(".json")])
        
        # Close the pool and wait for all processes to finish
        pool.close()
        pool.join()
    elif user_choice == 'n':
        # Ask the user for the specific config file name to run
        config_filename = input("Enter the config file name to run: ").strip()
        
        # Check if the specified config file exists
        if config_filename in config_files:
            process_config(config_filename)
        else:
            print(f"The config file '{config_filename}' does not exist in the 'configs' directory.")
    else:
        print("Invalid choice. Please enter 'all' or 'one'.")

# if __name__ == '__main__':

#     # config_filename = input("Enter config file name : ")

#     config_filename = "configs/config.json"
#     config = read_config(config_filename)

#     experiment_name = config["experiment_name"]
#     elitism = config["elitism"]
#     population_size = config["population_size"]
#     mutation_rate = config["mutation_rate"]
#     mutation_sigma = config["mutation_sigma"]
#     mutation_decay = config["mutation_decay"]
#     mutation_limit = config["mutation_limit"]
#     amount_optimization_steps = config["amount_optimization_steps"]
#     dna_bounds = config["dna_bounds"]
#     dna_start_position = config["dna_start_position"]
#     weight1 = config["weight1"]
#     weight2 = config["weight2"]
#     objective_fidelity = config["objective_fidelity"]
#     num_hops = config["num_hops"]
#     excel_file = config["excel_file"]
#     strategy = config["strategy"]

#     main_process(
#         experiment_name,
#         elitism,
#         population_size,
#         mutation_rate,
#         mutation_sigma,
#         mutation_decay,
#         mutation_limit,
#         amount_optimization_steps,
#         dna_bounds,
#         dna_start_position,
#         weight1,
#         weight2,
#         objective_fidelity,
#         num_hops,
#         excel_file,
#         strategy
#     )