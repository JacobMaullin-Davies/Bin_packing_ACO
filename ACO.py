"""
Bin packing problem using simple ACO. User can select from two problems:
BBP1 = 500 items of weights of i, where i equal to the item number (
eg item 2 = weight 2). These items are to be packed into 10 bins.
BPP2 = 500 items of weights of i^2, where i equal to the item number 2
(eg item 2 = weight 22 = 4). These items are to be packed into 50 bins
"""
import random
import numpy as np
import matplotlib.pyplot as plt

def generate_ant_paths(items, graph):
    """
    Generates ant path
    : param items: The item list
    : param graph: Graph
    : returns: List of bins chosen for each item placement
    """
    ant_paths = []
    for index, item in enumerate(items):
        ant_paths.append(choose_nextbin(index, graph))
    return ant_paths

def choose_nextbin(item_number, graph):
    """
    Weighted choice which bin the item should be placed into
    : param item_number: The item in the graph
    : param graph: Graph
    : returns: Chosen bin for the item placement
    """
    weights = []
    item_column = graph.get(item_number)
    total_pheromone = sum(item_column) #total pheromone for the bin column
    for path_pheromone in item_column:
        weights.append(path_pheromone / total_pheromone)
    path_choices = [i for i in range(len(item_column))]
    chosen_bin = random.choices(path_choices, weights) # randomly select a bin (path) to take, returns as a list
    return chosen_bin[0]

def evaluate_fitness(path, items, bins):
    """
    Evaluate the path chosen
    : param path: Ant path chosen
    : param items: Items to be placed in the bin
    : param bins: Number of bins
    : returns: Fitness of the path
    """
    bins = [0]*bins # list of bins with weights initialised at 0
    index = 0
    for bin_choice in path:
        bins[bin_choice] += items[index]
        index += 1
    return (max(bins) - min(bins))

def update_pheromone_fitness(ant_path, fitness_of_path, problem_graph):
    """
    The bin selected in the path is updated acording to the fitness
    : param ant_path: Ant path
    : param fitness_of_path: Path fitness
    : param problem_graph: Graph
    : returns: Updated Graph
    """
    for item_index, bin_index in enumerate(ant_path):
        problem_graph[item_index][bin_index] += 100/fitness_of_path
    return problem_graph

def evaporate_pheromone(graph, e_rate):
    """
    Mulitplies the enitre graph by evaporation value
    : param graph: Graph
    : param e_rate: Evaporation rate
    : returns: Updated graph
    """
    for item_number, pheromone_values in graph.items():
        graph[item_number] = [p*e_rate for p in pheromone_values]
    return graph

def bin_packing(number_bins, number_items, evaporation_rate, number_ants, item_weights):
    """
    Runs a full ACO path for each iteration
    : param number_bins: Number of bins
    : param items: Number of items
    : param evaporation_rate: Evaporation rate
    : param bins: Number of bins
    : param number_ants: Number of Ants
    : param item_weights: Weights of items
    : returns: End best fitness, Run fitness, Best Bin configuration, interation average
    """
    fitness_evaluations = 10000/number_ants
    tranversed_problem_graph = generate_construction_graph(number_bins, number_items)
    end_best_fitness, best_bin_config = number_bins**number_bins, item_weights[:1]*number_items
    global_best_fitness = end_best_fitness
    plt_fitness = []
    interation_avrg = []

    i = 0
    while i < fitness_evaluations:
        fitness_list, tranversed_problem_graph, bin_config_generations = ant_optimisation(number_ants,
        item_weights, number_bins, tranversed_problem_graph, evaporation_rate)
        i += 1
        for j in fitness_list:
            if global_best_fitness > j:#global best fitness
                global_best_fitness = j
                print("new global best:", global_best_fitness)
            plt_fitness.append(j)
        interation_avrg.append(sum(fitness_list)/len(fitness_list))

    index = 0
    for k in fitness_list: #the best path from the last iteration of ants
        if end_best_fitness > k:
            end_best_fitness = k
            best_bin_config = bin_config_generations[index]
        index += 1

    return end_best_fitness, plt_fitness, best_bin_config, interation_avrg

def ant_optimisation(number_ants, items, bins, problem_graph, evaporation_rate):
    """
    Ant run
    : param number_ants: Number of Ants
    : param items: Number of items
    : param bins: Number of bins
    : param problem_graph: Graph
    : param evaporation_rate: Evaporation rate
    : returns: Fitness list, Updated Graph, List of Ant paths
    """
    path_gen_list = []
    fitness_list = []
    for _ in range(number_ants): #run the number of ants to find paths
        path_gen = generate_ant_paths(items, problem_graph)#finds a path
        fitness_of_path = evaluate_fitness(path_gen, items, bins)#evaluate the fitness
        path_gen_list.append(path_gen)
        fitness_list.append(fitness_of_path)

    for i in range(number_ants): #update the graph for all the ant paths
        problem_graph = update_pheromone_fitness(path_gen_list[i], fitness_list[i], problem_graph)
        #update the graph according to the fitness
    problem_graph = evaporate_pheromone(problem_graph, evaporation_rate)
    #evapourate the whole graph
    return fitness_list, problem_graph, path_gen_list

def generate_construction_graph(bins, items):
    """
    Generates the constructed problme graph with randomly distributed pheromone
    : param bins: Number of bins
    : param bins: Number of items
    : returns: pheromone graph
    """
    random.seed() # change random selection of numbers for each new construction graph
    graph = {}
    for i in range(items):
        graph.update({i: [random.uniform(0,1) for _ in range(bins)]})
    return graph

def generate_item_weights(problem_def):
    """
    Generates the item weights for BBPX problem
    : param problem_def: problem BBPX input
    : returns: item weight array, Number of bins
    """
    item_weights = []
    if problem_def == 1:
        for i in np.arange(1, 501, 1):
            item_weights.append(i)
        bins = 10
    else:
        for i in np.arange(1, 501, 1):
            item_weights.append(i**2)
        bins = 50
    return item_weights, bins

def main_run(problem_input, p_input, e_input):
    """
    Main function run and output to terminal and graph plot.
    : param problem_input: problem BBPX input
    : pram p_input: Number of ants
    : e_input: Evaporation value
    """
    item_weights, bin_count = generate_item_weights(problem_input)

    optimal_fit = []
    run_fitness = []
    iteration_avrg_list = []

    print(len(item_weights), "items,", bin_count,
    "bins with p =", p_input, "e =", e_input)
    for _ in range(1): #can change the number of runs if desired
        optimal_fitness, fitness_overtime, bin_config, interation_avrg = bin_packing(bin_count, 500, e_input, p_input, item_weights)
        optimal_fit.append(optimal_fitness)
        run_fitness.append(fitness_overtime)
        iteration_avrg_list.append(interation_avrg)

    for i in range(len(optimal_fit)):
        print("Run ", i+1)
        print("End run optimal", optimal_fit[i])
        print("Minimium: ", min(run_fitness[i]))
        print("Maximum: ", max(run_fitness[i]))
        print("Average: ", (sum(iteration_avrg_list[i])/len(iteration_avrg_list[i])))

    bins_fill = [0]*bin_count
    index = 0
    for bin_choice in bin_config:
        bins_fill[bin_choice] += item_weights[index]
        index += 1
    #
    for g in range(len(bin_config)):
        bin_config[g] += 1
    #
    print("Item placemet for bin: ", bin_config)
    print("Bins total: ", bins_fill)
    #
    plt.title("Average Fitness overtime")
    plt.xlabel('Number of iterations')
    plt.ylabel('Mean Fitness')

    run = 0
    for i in iteration_avrg_list:
        plt.plot(range(len(iteration_avrg_list[run])), i)
        run += 1
    plt.show()

if __name__ == "__main__":
    print("Runtime average: BBP1 = 30 seconds. BBP2 = 90 seconds")
    VALID = False
    while VALID is False:
        problem_input = int(input("Enter '1' for BBP1 OR '2' for BBP2: "))
        if problem_input in (1,2):
            p_input = int(input("Enter the number of Ants > 0 (e.g. 10): "))
            e_input = float(input("Enter the evaporation value between 0 and 1 (e.g. 0.6): "))
            if p_input > 0 and e_input > 0 and e_input < 1:
                VALID = True
            else:
                print("Not valid inputs")
        else:
            print("Not valid input")

    main_run(problem_input, p_input, e_input)
