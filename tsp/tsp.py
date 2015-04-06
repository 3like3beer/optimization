#!/usr/bin/python
'''
Simplest OpenOpt TSP example;
requires networkx (http://networkx.lanl.gov)
and FuncDesigner installed.
For some solvers limitations on time, cputime, "enough" value, basic GUI features are available.
See http://openopt.org/TSP for more details
'''

import math
from collections import namedtuple
from openopt import *
import networkx as nx

Point = namedtuple("Point", ['x', 'y'])


def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


def tour_length(node_count, points, sol):
    # calculate the length of the tour
    obj = length(points[sol[-1]], points[sol[0]])
    for index in range(0, node_count - 1):
        obj += length(points[sol[index]], points[sol[index + 1]])

    return obj


def oo_solution(points, node_count):

    G = nx.Graph()
    G.add_edges_from(\
                     [(i,j,{'time': length(pi,pj), 'cost':length(pi,pj)}) for (i,pi) in enumerate(points) for (j,pj) in enumerate(points) if i != j ])

    # default objective is "weight"
    # parameter "start" (node identifier, number or string) is optional
    p = TSP(G, objective = 'time', start = 0) #, [optional] returnToStart={True}|False, constraints = ..., etc
    #p.solve()
    r = p.solve('lpSolve') # also you can use some other solvers - sa, interalg, OpenOpt MILP solvers

    # if your solver is cplex or interalg, you can provide some stop criterion,
    # e.g. maxTime, maxCPUTime, fEnough etc, for example
    # r = p.solve('cplex', maxTime = 100, fEnough = 10)
    # i.e. stop if solution with 10 nodes has been obtained or 100 sec were elapsed
    # (gplk and lpSolve has no iterfcn connected and thus cannot handle those criterions)

    # also you can use p.manage() to enable basic GUI (http://openopt.org/OOFrameworkDoc#Solving)
    # it requires tkinter installed, that is included into PythonXY, EPD;
    # for linux use [sudo] easy_install tk or [sodo] apt-get install python-tk
    #r = p.manage('cplex')

    #print(r.nodes)
    #print(r.edges)
    #print(r.Edges)
    #solution =
    return ([r.nodes[i] for i in range(0,node_count)])



def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    node_count = int(lines[0])

    points = []
    for i in range(1, node_count+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    solution = oo_solution(points,node_count)
    print(solution)

    obj = tour_length(node_count, points, solution)

    # prepare the solution in the specified output format
    output_data = str(obj) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data




if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        input_data_file = open(file_location, 'r')
        input_data = ''.join(input_data_file.readlines())
        input_data_file.close()
        print solve_it(input_data)
    else:
        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)'

