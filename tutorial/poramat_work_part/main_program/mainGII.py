# from ga_DevelopI import *
# from all_function import *
import sys

sys.path.insert(0, '/Users/poramat/Documents/qwanta/tutorial/Genetic_algorithm_development/')

print('import success')

# def main_process(
#     experiment_name,
#     elitism = 0.2,
#     population_size = 500,
#     mutation_rate = 0.8,
#     mutation_sigma = 0.3,
#     mutation_decay = 0.999,
#     mutation_limit = 0.01,
#     amount_optimisation_steps = 400,
#     dna_bounds = (-5.11, 5.11),
#     dna_start_position = [4.8, 4.8]
# ):    

#     ga = GA_version_2(
#         dna_size=len(dna_start_position),
#         dna_bounds=dna_bounds,
#         dna_start_position=dna_start_position,
#         elitism=elitism,
#         population_size=population_size,
#         mutation_rate=mutation_rate,
#         mutation_sigma=mutation_sigma,
#         mutation_decay=mutation_decay,
#         mutation_limit=mutation_limit,
#     )

#     first_generation = ga.population

#     # photon loss (db/km): 
#     photon_loss = [ga.population[i][0] for i in range(ga.population_size)]
#     # gate error: (%)
#     gate_error = [ga.population[i][1] for i in range(ga.population_size)]
#     # measurement error: (%)
#     measurement_error = [ga.population[i][2] for i in range(ga.population_size)]
#     # coherence time (seconds):
#     coherence = [ga.population[i][3] for i in range(ga.population_size)]

#     for step in range(amount_optimisation_steps):
#         pass