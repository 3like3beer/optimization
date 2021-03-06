#!/usr/bin/python
# -*- coding: utf-8 -*-

import pulp
from itertools import combinations_with_replacement,product
from collections import namedtuple
import collections
import functools
Item = namedtuple("Item", ['index', 'value', 'weight'])
Item2 = namedtuple("Item", ['index', 'value', 'weight','vpw'])
Solution= namedtuple("Solution", ['nb_items', 'capacity','taken', 'value','weight'])

max_value = 0
selected =[]


def get_max_value():
    return max_value

def get_selected_taken():
   return selected

def update_max_value(value,taken):
    global max_value
    global selected
    if value> max_value:
        print(str(max_value))
        max_value = value
        selected = taken[:]
        print("selected " + str(selected))
   
def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    # a trivial greedy algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    value,weight,taken = pulp_solve(items,capacity)
    
    
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data

# a trivial greedy algorithm for filling the knapsack
# it takes items in-order until the knapsack is full
def pulp_solve(items,capacity):
    knapsack = pulp.LpProblem("Knapsack Model", pulp.LpMaximize)
    x = [pulp.LpVariable("x"+str(it.index), 0, 1, 'Integer') for it in items]
        
    objective = pulp.LpAffineExpression([ (x[i.index],i.value) for i in items])
    knapsack.setObjective(objective)
    #knapsack += sum([items.value[i]*x[i] for i in items])
    knapsack += sum([i.weight*x[i.index] for i in items]) <= capacity -5
    knapsack.solve(pulp.PULP_CBC_CMD())
    taken = [int(i.value()) for i in x]
    value = sum([items[i].value*t for (i,t) in enumerate(taken)])
    weight = sum([items[i].weight*t  for (i,t) in enumerate(taken)])
    print(weight)
    return value,weight,taken
    
def trivial_algo(items,capacity):
    value = 0
    weight = 0
    taken = [0]*len(items)

    for item in items:
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight
    return value,weight,taken

        
def dfs(graph,root):
    visited = [False for i in graph.vertices]
    visited[root] = True
    for v in root.adjacents:
        if not(visited[v]):
            dfs(graph,v)
            
def bfs(graph,root):
    visited = [False for i in graph.vertices]
    q=[]
    q.enqueue(root);
    visited[root] = True;
    while not(q.isEmpty()):
        v = q.dequeue();
        for w in graph.adj(v):
            if not(visited[w]):
                q.enqueue(w)
                visited[w] = True
                #edgeTo[w] = v        
        
def dfs_knapsack(sorted_items,capacity,taken):
    global selected
    if len(taken)<len(sorted_items):
        taken = taken + [0]
        for v in [1,0]:
            taken[len(taken)-1]=v
            if is_feasible(taken,sorted_items,capacity):
                current_value,current_weight = get_value(taken,sorted_items,capacity)
                current_up_bound,current_weight2,taken = get_upper_bound0(current_value,current_weight,taken,sorted_items,capacity)
                if current_value>get_max_value():
                    update_max_value(current_value,taken)
                    #print("current_up_bound " + str (current_up_bound))
                if current_up_bound > get_max_value() + 0:
                    dfs_knapsack(sorted_items,capacity,taken)
    return get_max_value(),get_selected_taken()
    
def bnb_algo(items,capacity):
    global max_value
    global selected
    taken =[]
    update_max_value( 0,taken)
    sorted_items = sort_items(items)
    dfs_knapsack(sorted_items,capacity,taken)
    pad = len(sorted_items) - len(selected) 
    taken = selected
    if pad>0:
        for i in range(pad):
            taken = taken + [0]
    taken_final = [0 for t in taken]
    for t,item in zip(taken,sorted_items):
        if t>0:
            #print(item.index)
            taken_final[item.index] = 1
    return max_value,0,taken_final  
    
def get_upper_bound0(current_value,current_weight,taken,sorted_items,capacity):
    for item in sorted_items[len(taken):]:
        if current_weight + item.weight <= capacity:
            current_value += item.value
            current_weight += item.weight
        else:
            part = (capacity - current_weight) / item.weight
            current_value += (item.value) * part
            #weight += (item.weight) * taken[item.index] #=capa
            current_weight = capacity
            return current_value,current_weight,taken
    return current_value,current_weight ,taken
    
def get_upper_bound(items,capacity):
    value = 0
    weight = 0
    taken = [0]*len(items)
    for item in items:
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight
        else:
            taken[item.index] = (capacity - weight) / item.weight
            value += (item.value) * taken[item.index]
            #weight += (item.weight) * taken[item.index] #=capa
            weight = capacity
            return value,weight,taken
    return value,weight,taken
    
def is_feasible(taken,items,capacity):
    total_weight = sum([t * item.weight for (t,item) in zip(taken,items)])
    return (total_weight<capacity)

def get_value(taken,items,capacity):
    total_value = sum([t * item.value for (t,item) in zip(taken,items)])
    total_weight = sum([t * item.weight for (t,item) in zip(taken,items)])
    return total_value,total_weight

class memoized(object):
   '''Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
  '''
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args):
      if not isinstance(args, collections.Hashable):
         # uncacheable. a list, for instance.
         # better to not cache than blow up.
         return self.func(*args)
      if args in self.cache:
         return self.cache[args]
      else:
         value = self.func(*args)
         self.cache[args] = value
         return value
   def __repr__(self):
      '''Return the function's docstring.'''
      return self.func.__doc__
   def __get__(self, obj, objtype):
      '''Support instance methods.'''
      return functools.partial(self.__call__, obj)
      
def knapsack(items, maxweight):
    """
    Solve the knapsack problem by finding the most valuable
    subsequence of `items` subject that weighs no more than
    `maxweight`.

    `items` is a sequence of pairs `(value, weight)`, where `value` is
    a number and `weight` is a non-negative integer.

    `maxweight` is a non-negative integer.

    Return a pair whose first element is the sum of values in the most
    valuable subsequence, and whose second element is the subsequence.

    >>> items = [(4, 12), (2, 1), (6, 4), (1, 1), (2, 2)]
    >>> knapsack(items, 15)
    (11, [(2, 1), (6, 4), (1, 1), (2, 2)])
    """

    # Return the value of the most valuable subsequence of the first i
    # elements in items whose weights sum to no more than j.
    @memoized
    def bestvalue(i, j):
        if i == 0: return 0
        value, weight = items[i - 1]
        if weight > j:
            return bestvalue(i - 1, j)
        else:
            return max(bestvalue(i - 1, j),
                       bestvalue(i - 1, j - weight) + value)

    j = maxweight
    result = []
    taken = [0]*len(items)
    for i in range(len(items), 0, -1):
        if bestvalue(i, j) != bestvalue(i - 1, j):
            result.append(items[i - 1])
            taken[i-1]=1
            j -= items[i - 1][1]
    result.reverse()
    return bestvalue(len(items), maxweight), result #,taken
    
def dp_algo2(items,capacity):
    new_items= [(v.value,v.weight) for v in items]    
    best_value,items,taken = knapsack(new_items,capacity)
    return best_value,0,taken 
    
def dp_algo(items,capacity):
    solutions = []
    #Tri les items par densitÃ© de valeur
    sorted_items = sort_items(items)
    taken = [0]*len(sorted_items)
    max_value = 0
    weight = 0 
    last_taken = 0
    considered_items = 1
    #Ajoute les items 1 par 1
    while considered_items < len(sorted_items):
        next_item = sorted_items[considered_items-1]
        #s'il tient on prend
        if weight + next_item.weight   < capacity:
            weight = weight + next_item.weight 
            taken[next_item.index]=1
            last_taken = next_item.index
            max_value = max_value + next_item.value
            solutions.append(Solution(considered_items, capacity, taken,max_value,weight))
        else:
            #tant qu'il ne tient pas, on backtrack les derniers pris
            previous_solution= solutions[last_taken]
            #s'il tient toujours pas
            if previous_solution.weight + next_item.weight < capacity:
                #remove previous item
                weight = previous_solution.weight + next_item.weight 
                taken[next_item.index]=1
                max_value = max_value + next_item.value
                solutions.append(Solution(considered_items, capacity, taken,max_value,weight))
                    
        considered_items +=1
    return max_value,weight,taken 
    


def dp_recurse(items,capacity,taken,value,weight):
    
    if len(items)==0:
        return value,weight,taken
    else:
        item = items.pop()
        value_if_not_selected = 0
        if item.weight < capacity:
            value_if_selected = item.value + dp_algo(items,capacity-item.weight)
            value_if_not_selected = dp_recurse(items,capacity)
            if value_if_not_selected>value_if_selected:
                taken[item.index]=0
                weight = weight+item.weight
                value= value_if_not_selected
            else:
                taken[item.index]=0
                weight = weight+item.weight
                value= value_if_not_selected
        else:                
            taken[item.index]=0
            weight = weight+item.weight
            value= value_if_not_selected

def sort_items(items):
    sorted_items = [Item2(item.index,item.value,item.weight,(0.0+item.value)/(0.0+item.weight)) for item in items]
    sorted_items.sort(reverse=True,key=lambda tup: tup[3])
    return sorted_items
    
def greedy_algo(items,capacity):
    sorted_items = sort_items(items)
    
    value = 0
    weight = 0
    taken = [0]*len(items)

    for item in sorted_items:
        if weight + item.weight <= capacity:
            #print(item.index)
            taken[item.index] = 1
            value += item.value
            weight += item.weight
    return value,weight,taken
        
import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        input_data_file = open(file_location, 'r')
        input_data = ''.join(input_data_file.readlines())
        input_data_file.close()
        print (solve_it(input_data))
    else:
        print ('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

