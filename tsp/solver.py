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

    node_count = int(lines[0])

    points = []
    for i in range(1, node_count+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    solution = pulp_solution(points,node_count)
    print(len(solution))
    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, node_count-1):
        obj += length(points[solution[index]], points[solution[index+1]])

    # prepare the solution in the specified output format
    output_data = str(obj) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data

def trivial_solution(points,node_count):
    # build a trivial solution
    # visit the nodes in the order they appear in the file
    return range(0, node_count)

def build_variable(node_in,node_out):
    if node_in > node_out:
        return pulp.LpVariable("x_in" + str(node_in) + "_out" + str(node_out) , 0,1, 'Binary')
    else:
        return 0.0

def find_out_node(node_in,out_edges,node_set,visited):
    #print ("input " + str(node_in))
    for node_out in node_set:
        if out_edges[node_in][node_out] != 0.0:
            if not(visited[node_in][node_out]):
                if out_edges[node_in][node_out].value()>0.7:
                    #print ("out1 " + str(out_edges[node_in][node_out]))
                    return node_out,1

        if out_edges[node_out][node_in] != 0.0:
            if not(visited[node_out][node_in]):
                if out_edges[node_out][node_in].value()>0.7:
                    #print ("out2 " + str(out_edges[node_out][node_in]))
                    return node_out,2
    return None,0

def pulp_solution(points,node_count):
    tsp = pulp.LpProblem("Tsp Model", pulp.LpMinimize)
    node_set = range(0,node_count)
    out_edges =  [[build_variable(node_in,node_out) for node_in in node_set] for node_out in node_set]
    #objective = pulp.LpAffineExpression([ (out_edges[i,k],i.value) for i in node_set for k in node_set))
    #out_edges2 =  [[1 for node_in in node_set] for node_out in node_set]
    #print([out_edges2[node_out] for node_out in node_set])
    #print([[out_edges2[node_in][node_out]*length(points[node_in], points[node_out]) for node_in in node_set ] for node_out in node_set])
    tsp+= sum([sum([out_edges[node_in][node_out]*length(points[node_in], points[node_out]) for node_in in node_set ]) for node_out in node_set])

    for node in node_set:
        tsp += sum(out_edges[node][v] for v in node_set) + sum(out_edges[v][node] for v in node_set) == 2

    tsp.solve()

    for t in tsp.variables():
        if t.value()>0.5:
            print (str(t) + " " + str(t.value()))

    out = [0]
    visited = [[False for node_in in node_set] for node_out in node_set]
    node_in = 0
    while node_in is not None:
        node_out,a = find_out_node(node_in,out_edges,node_set,visited)
        if node_out is not None:
            out.append(node_out)
            if a == 1:
                visited[node_in][node_out] = True
                #print ("visited1 " + str(node_in) + " " +str(node_out))
            else:
                visited[node_out][node_in] = True
                #print ("visited2 " + str(node_out) + " "  + str(node_in))
        node_in = node_out
    print(out)
    print(sorted(out))
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

