#!/usr/bin/python
# -*- coding: utf-8 -*-
from ImageOps import colorize
from operator import is_
from reportlab.lib.validators import isInstanceOf

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
    pulp_solve(node_count,edges)
    # prepare the solution in the specified output format
    output_data = str(node_count) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data

def pulp_solve(node_count,edges):
    coloring = pulp.LpProblem("Color Model", pulp.LpMinimize)
    color_set = range(0,node_count)
    is_color =  [[pulp.LpVariable("x_col" + str(color) + "_node" + str(node) , 0,1, 'Binary') for color in color_set] for node in color_set]
    obj = pulp.LpVariable("objective",0,node_count,'Integer')
    objective = pulp.LpAffineExpression(obj)
    coloring.setObjective(objective)

    for color in color_set:
        for node in color_set:
            coloring += node * is_color[color][node] <= obj
        coloring += sum(is_color[color][v] for v in color_set) == 1
        for e in edges:
            coloring += is_color[color][e[0]] + is_color[color][e[1]] <= 1
    print(coloring)
    coloring.solve(pulp.COIN_CMD())

    for color in color_set:
        for node in color_set:
            print(is_color[color][node])
            print(is_color[color][node].value())
    return 0

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

