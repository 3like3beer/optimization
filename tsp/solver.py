#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import pulp
from collections import namedtuple

Point = namedtuple("Point", ['x', 'y'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    # build a trivial solution
    # visit the nodes in the order they appear in the file
    solution = range(0, nodeCount)

    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount-1):
        obj += length(points[solution[index]], points[solution[index+1]])

    # prepare the solution in the specified output format
    output_data = str(obj) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


def pulp_solution(points,node_count):
    tsp = pulp.LpProblem("Tsp Model", pulp.LpMinimize)
    node_set = range(0,node_count)
    out_edges =  [[pulp.LpVariable("x_col" + str(color) + "_node" + str(node) , 0,1, 'Binary') for color in range(0,node)] for node in node_set]
    #objective = pulp.LpAffineExpression([ (out_edges[i,k],i.value) for i in node_set for k in node_set))
    tsp+= sum([sum([out_edges[i,j]*length(points(i), points(j)) for i in range(0,j)]) for j in node_set])

    for color in node_set:
        for node in node_set:
            tsp += node * out_edges[color][node] <= obj
        tsp += sum(out_edges[color][v] for v in node_set) == 1
        for e in edges:
            tsp += out_edges[e[0]][color] + out_edges[e[1]][color] <= 1
    #print(coloring)
    tsp.solve(pulp.GLPK_CMD())

    out = []
    for node in node_set:
        for color in node_set:
            if out_edges[node][color].value()>0.5:
                #print ("col" + str(color) + "node" + str(node))
                out.append(color)
    #print out
    return out

import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        input_data_file = open(file_location, 'r')
        input_data = ''.join(input_data_file.readlines())
        input_data_file.close()
        print solve_it(input_data)
    else:
        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)'

