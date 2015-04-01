#!/usr/bin/python
# -*- coding: utf-8 -*-

import pulp

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    # build a trivial solution
    # every node has its own color
    solution = range(0, node_count)

    # prepare the solution in the specified output format
    output_data = str(node_count) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data

def pulp_solve(node_count,edges):
    coloring = pulp.LpProblem("Color Model", pulp.LpMinimize)
    potential_colors = range(0,node_count)
    color(node,col) =
    x = [pulp.LpVariable("x"+str(it.index), 0, 1, 'Integer') for it in items]

    objective = pulp.LpAffineExpression([ (x[i.index],i.value) for i in items])
    coloring.setObjective(objective)
    coloring += sum([i.weight*x[i.index] for i in items]) <= capacity -5
    coloring.solve(pulp.COIN_CMD())
    taken = [int(i.value()) for i in x]
    value = sum([items[i].value*t for (i,t) in enumerate(taken)])
    weight = sum([items[i].weight*t  for (i,t) in enumerate(taken)])
    print(weight)
    return value,weight,taken

import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        input_data_file = open(file_location, 'r')
        input_data = ''.join(input_data_file.readlines())
        input_data_file.close()
        print solve_it(input_data)
    else:
        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)'

