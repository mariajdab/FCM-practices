import random
import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from fcmpy import FcmSimulator
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import time
from pandas import read_csv

def read_csv_data(path):
    df = pd.read_csv(path)
    return df


#%%Define the objective function with sigmoid
def funcion_objetivo(prescriptive_concepts, individuo):        
    for i in range(len(prescriptive_concepts)):
        if prescriptive_concepts[i] > 1 or prescriptive_concepts[i] < 0:
            return 6,
    
    weight_mat = np.genfromtxt("fcmm_weight_vino.csv", delimiter=',')

    concept_names = ['c' + str(x) for x in range(1, len(individuo) + 1)]

    weight_matrix = pd.DataFrame(weight_mat, columns = concept_names)

    individuo_dict = dict(zip(concept_names, individuo))
    classical_fcm_results = FcmSimulator.simulate(initial_state = individuo_dict,
                                weight_matrix = weight_matrix,
                                transfer = 'sigmoid', 
                                inference = 'mKosko', 
                                thresh = 0.001,
                                iterations = 120,
                                l = 10)
    
    
    initial_vector = individuo[0:5] + [prescriptive_concepts[0]] + [individuo[6]] + [prescriptive_concepts[1]] + individuo[8:]

    initial_vector_dict = dict(zip(concept_names, initial_vector))
    results = FcmSimulator.simulate(initial_state = initial_vector_dict,
                                weight_matrix = weight_matrix,
                                transfer = 'sigmoid', 
                                inference = 'mKosko', 
                                thresh = 0.001,
                                iterations = 120,
                                l = 10)
    

    dataset_stable_state_vector = classical_fcm_results.tail(1).values.flatten().tolist()
    target = [dataset_stable_state_vector[11]]
    system_concepts = dataset_stable_state_vector[0:5] + [dataset_stable_state_vector[6]] + dataset_stable_state_vector[8:11]   + target
    
    
    prv_fcm_stable_state_vector = results.tail(1).values.flatten().tolist()
    target_prv_fcm = [prv_fcm_stable_state_vector[11]]
    systems_concepts_prv_fcm =  prv_fcm_stable_state_vector[0:5] + [prv_fcm_stable_state_vector[6]] + prv_fcm_stable_state_vector[8:11] + target_prv_fcm
    
    
    system_concepts_difference = [abs(e1 - e2) for e1, e2 in zip(system_concepts,systems_concepts_prv_fcm)]
    fitness = sum(system_concepts_difference)
    return fitness,


def setup(individuo):
    #%%Read the weight matrix and initial vector
    weight_mat = np.genfromtxt("fcmm_weight_vino.csv", delimiter = ",")
    #%%
    num_concepts = weight_mat.shape[0]
    #Number of concepts of the system
    num_system_concepts = 5
    #Number of prescriptive concepts
    num_prescriptive_concepts = 2
    creator.create("FitnessMin", base.Fitness, weights = (-1.0,))
    creator.create("Individual", list, fitness = creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_uniform", random.uniform, 0, 1) 
    toolbox.register("generate_prescriptive_concepts", tools.initRepeat,
                    creator.Individual, toolbox.attr_uniform, 
                    num_prescriptive_concepts)
    #%%
    prescriptive_concepts = toolbox.generate_prescriptive_concepts()
    #Print the content of individuo
    print(prescriptive_concepts, "aqui")
    #%% Create a population function to generate the population
    toolbox.register("population", tools.initRepeat, list, 
                    toolbox.generate_prescriptive_concepts, 150)
    toolbox.population()
    #%% Register the objective function with sigmoid and other functions
    #Register objective function
    toolbox.register("evaluate", funcion_objetivo, individuo=individuo)
    #%%
    #Register the cross function
    toolbox.register("mate", tools.cxOnePoint)
    #Register the mutation function
    #mu is the mean
    #sigma is standard deviation
    #indpb is the mutation probability
    toolbox.register("mutate", tools.mutGaussian, mu = 0, sigma = 5, indpb = 0.1)
    toolbox.register("select", tools.selTournament, tournsize = 3)
    #%% Test the objective functions
    print("Fitness: %s" %toolbox.evaluate(prescriptive_concepts))
    return toolbox


#%%Define the main
#In this case we use the EaSimple algorithm
def main(toolbox):
    cxpb, mutpb, ngen = 0.50, 0.50, 20#define the probabilities and generations number
    pop = toolbox.population() #generate the population
    hof = tools.HallOfFame(1) #object that will save the best solution
    stats = tools.Statistics(lambda ind: ind.fitness.values) #Save the registered statistics
    stats.register("avg", np.mean) #Register the mean
    stats.register("std", np.std) #Register the standar deviation
    stats.register("min", np.min) #Register the mean
    stats.register("max", np.max) #Register the max
    logbook = tools.Logbook() #Register the evolution
    #Define the algorithm EaSimple
    #verbose is for showing stats in each iteration
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb = cxpb,
                                       mutpb = mutpb, ngen = ngen,
                                       stats = stats,
                                       halloffame = hof,
                                       verbose = True)
    return hof, logbook


if __name__ == "__main__":
    df = read_csv('test_wine_final_data.csv', delimiter = ",", header=None, index_col=False)
    individuos = df.astype('float64').values.tolist()

    for individuo in individuos:
        t1 = time.time()
        pool = multiprocessing.Pool(processes = 4)
        toolbox = setup(individuo)
        toolbox.register("map", pool.map)
        best, log = main(toolbox)
        mejor_individuo = pd.DataFrame(best[0])
        mejor_individuo2 = mejor_individuo.transpose()
        mejor_fitness = pd.DataFrame(best[0].fitness.values)
        final_result = pd.concat([mejor_individuo2, mejor_fitness], axis = 1)
        t2 = time.time()
        print("Tiempo de ejecución con paralelización %f" %(t2-t1))
        final_result.to_csv("wine_valores_prescriptivos.csv", header=None, index = False, mode='a')
    

   