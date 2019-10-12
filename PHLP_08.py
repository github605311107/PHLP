# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time

time_start = time.time()
def loadData():
    cl = []
    wl = []
    c = np.zeros((n,n))
    w = np.zeros((n,n))
    coordinate = np.zeros((n,2))
    fr = pd.read_csv("CAB.csv")
    for row in fr.itertuples():
        cl.append(row[1])
        wl.append(row[2])
    index = 0
    for i in range(0,n):
        for j in range(0,n):
            c[i][j] = cl[index]
            w[i][j] = wl[index]
            index = index + 1
    
    fcoordinate = pd.read_csv("coordinate.csv")
    j = 0
    for row in fcoordinate.itertuples():
        for i in range(2):
            coordinate[j][i] = row[i+2]        
        j = j + 1
    return c,w,coordinate

def generate_initial_population(c,w):
    population_list = []
    hub_list = random.sample(list(range(0,25)),4)   # hub from [0,1,2...,24]
    for i in range(pop_size):
        chromosome = generate_chromosome(hub_list)
        cost = cost_function(chromosome,c,w,coordinate)
        population_list.append([chromosome,hub_list,cost])
    return population_list

def generate_chromosome(hubs):
    chromosome = []
    n2h_cost = []
    for i in range(25):
        chromosome.append(None)
    for i in range(25):
        for j in hubs:
            n2h_cost.append(c[i][j])
        min_cost = min(n2h_cost)
        min_index = n2h_cost.index(min_cost)
        chromosome[i] = hubs[min_index]
        n2h_cost = []
    return chromosome

def cost_function(chromosome,c,w,coordinate):
    dist = np.zeros((n,n))
    cost_list = []
    for i in range(n):
        for j in range(n):
            node1_x = int(coordinate[i][0])
            node1_y = int(coordinate[i][1])
            node2_x = int(coordinate[j][0])
            node2_y = int(coordinate[j][1])
            d = np.sqrt(pow((node1_x - node2_x),2) + pow((node1_y - node2_y),2))
            dist[i][j] = d
    
    for i in range(n):
        cost_collection = c[i][chromosome[i]] * w[i][chromosome[i]] * dist[i][chromosome[i]]
        cost_distribution = c[chromosome[i]][i] * w[chromosome[i]][i] * dist[chromosome[i]][i]
        cost_list.append(cost_collection + cost_distribution)
    cost = sum(cost_list)
    return cost

def natural_selection(population_list):
    fitvalue_list = []
    select_list = []
    temp = 0.0
    for item in population_list:
        fitvalue_list.append((1.0 / item[2]) + temp)
        temp = (1.0 / item[2]) + temp
    for i in range(pop_size):
        fitvalue_list[i] = fitvalue_list[i] / fitvalue_list[pop_size - 1]
    random_list = sorted([random.random() for i in range(pop_size)])
    for i in range(pop_size):
        if fitvalue_list[i] > random_list[i]:
            select_list.append(i)
        else:
            for j in range(i,pop_size):
                if fitvalue_list[j] > random_list[i]:
                    select_list.append(j)
                    break
    return select_list
"""
def crossover(pop_1,pop_2):
    random_chromosome_point = np.random.randint(0,n)
    random_hubs_point = np.random.randint(0,p)
    swap = [-1 for i in range(random_chromosome_point)]
    for i in range(random_chromosome_point):
        swap[i] = pop_1[0][i]
        pop_1[0][i] = pop_2[0][i]
        pop_2[0][i] = swap[i]
    swap = [-1 for i in range(random_hubs_point)]
    for i in range(random_hubs_point):
        swap[i] = pop_1[1][i]
        pop_1[1][i] = pop_2[1][i]
        pop_2[1][i] = swap[i]
    n2h_cost = []
    for i in range(len(pop_1[0])):
        if pop_1[0][i] not in pop_1[1]:
            for j in pop_1[1]:
                n2h_cost.append(c[i][j])
            min_cost = min(n2h_cost)
            min_index = n2h_cost.index(min_cost)
            pop_1[0][i] = pop_1[1][min_index]
    n2h_cost = []
    for i in range(len(pop_2[0])):
        if pop_2[0][i] not in pop_2[1]:
            for j in pop_2[1]:
                n2h_cost.append(c[i][j])
            min_cost = min(n2h_cost)
            min_index = n2h_cost.index(min_cost)
            pop_2[0][i] = pop_2[1][min_index]
    cost_pop_1 = cost_function(pop_1[0],c,w,coordinate)
    cost_pop_2 = cost_function(pop_2[0],c,w,coordinate)
    pop_1[2] = cost_pop_1
    pop_2[2] = cost_pop_2
    return pop_1,pop_2
"""
def crossover(pop_1,pop_2):
    hubs_A_coordinate = []
    hubs_B_coordinate = []
    D_dist = np.zeros((p,p)).tolist()
    for index in pop_1[1]:
        hubs_A_coordinate.append(coordinate[index])
    for index in pop_2[1]:
        hubs_B_coordinate.append(coordinate[index])
    for i in range(p):
        for j in range(p):
            node1_x = int(hubs_A_coordinate[i][0])
            node1_y = int(hubs_A_coordinate[i][1])
            node2_x = int(hubs_B_coordinate[j][0])
            node2_y = int(hubs_B_coordinate[j][1])
            d = np.sqrt(pow((node1_x - node2_x),2) + pow((node1_y - node2_y),2))
            D_dist[i][j] = d
    
    hubs_A1 = []
    hubs_B1 = []
    for i in range(p):
        min_dist = min(D_dist[i])
        min_index = D_dist[i].index(min_dist)
        hubs_B1.append(pop_1[1][min_index])
    for j in range(p):
        one_list=[]
        for i in range(p):
            one_list.append(D_dist[i][j])
        min_dist = min(one_list)
        min_index = one_list.index(min_dist)
        hubs_A1.append(pop_2[1][min_index])
    
    chromosome_A1 = generate_chromosome(hubs_A1)
    chromosome_B1 = generate_chromosome(hubs_B1)
    chromosome_A = pop_1[0]
    chromosome_B = pop_2[0]
    offspring_chromosome1,huns1 = crossover_comparison(chromosome_A,chromosome_B1)
    offspring_chromosome2,huns2 = crossover_comparison(chromosome_A1,chromosome_B)
    cost_offspring1 = cost_function(offspring_chromosome1,c,w,coordinate)
    cost_offspring2 = cost_function(offspring_chromosome2,c,w,coordinate)
    pop_1 = [offspring_chromosome1,huns1,cost_offspring1]
    pop_2 = [offspring_chromosome2,huns2,cost_offspring2]
    return pop_1,pop_2
    
def crossover_comparison(chromosome_A,chromosome_B1):
    offspring_chromosome = []
    break_num = 20
    while break_num > 0:
       for index in range(len(chromosome_A)):
           if chromosome_A[index] == chromosome_B1[index]:
               offspring_chromosome.append(chromosome_A[index])
           else:
               if random.randint(0,1) == 0:
                   offspring_chromosome.append(chromosome_A[index])
               else:
                   offspring_chromosome.append(chromosome_B1[index])
       hubs = []
       for gen in offspring_chromosome:
           if gen not in hubs:
               hubs.append(gen)
       if len(hubs) == p:
           return offspring_chromosome,hubs
       else:
           print("尝试再次杂交...")
           break_num = break_num - 1
    print("本轮杂交失败，程序将停止！！！")
    
def mutation(pop):
    random_mutation = random.random()
    if random_mutation < mutation_probability:
        random_hub = random.choice(pop[1])
        random_gen = random.randint(0,n - 1)
        pop[0][random_gen] = random_hub
    pop[2] = cost_function(pop[0],c,w,coordinate)
    return pop

def generate_offspring():
    select_list = natural_selection(population_list)
    natural_population_list = [None for i in range(pop_size)]
    offspring_list = [None for i in range(pop_size)]
    for i in range(pop_size):
        natural_population_list[i] = population_list[select_list[i]]
    for i in range(pop_size):
        if i == pop_size - 1:
            i_next = 0
            crossover_pop1,crossover_pop2 = crossover(natural_population_list[i],natural_population_list[i_next])
            mutation_pop1 = mutation(crossover_pop1)
            mutation_pop2 = mutation(crossover_pop2)
            pops_cost = [natural_population_list[i][2],mutation_pop1[2],mutation_pop2[2]]
            min_cost = min(pops_cost)
            min_index = pops_cost.index(min_cost)
            if min_index == 0:
                offspring_list[i] = natural_population_list[i]
            elif min_index ==1:
                offspring_list[i] = mutation_pop1
            else:
                offspring_list[i] = mutation_pop2
        else:
            crossover_pop1,crossover_pop2 = crossover(natural_population_list[i],natural_population_list[i+1])
            mutation_pop1 = mutation(crossover_pop1)
            mutation_pop2 = mutation(crossover_pop2)
            pops_cost = [natural_population_list[i][2],mutation_pop1[2],mutation_pop2[2]]
            min_cost = min(pops_cost)
            min_index = pops_cost.index(min_cost)
            if min_index == 0:
                offspring_list[i] = natural_population_list[i]
            elif min_index == 1:
                offspring_list[i] = mutation_pop1
            else:
                offspring_list[i] = mutation_pop2
    return offspring_list

def data_show(best_list):
    plt.plot(best_list)
    plt.show()

p = 4
n = 25
pop_size = 200
mutation_probability = 0.2
termination_condition = 300
c,w,coordinate = loadData()
population_list = generate_initial_population(c,w)
best_pop = population_list[0]
best_list = []
for pop in population_list:
    if best_pop[2] > pop[2]:
        best_pop = pop
while termination_condition > 0:
    offspring_list = generate_offspring()
    population_list = offspring_list
    for i in range(len(population_list)):
        if population_list[i][2] < best_pop[2]:
            best_pop = population_list[i]
    print(best_pop[2])
    print("termination_condition:%s"%termination_condition)
    best_list.append(best_pop[2])
    termination_condition = termination_condition - 1
time_end = time.time()   
data_show(best_list)    
print("Obj:%s"%best_pop[2])
print("Assignment:%s"%best_pop[0])
print("Hubs:%s"%best_pop[1])
run_time = time_end - time_start
print("run time:%s"%run_time)

