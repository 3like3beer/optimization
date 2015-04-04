#!/usr/bin/python
# -*- coding: utf-8 -*-

import random
import math
from reportlab.lib.colors import toColor
from __builtin__ import enumerate
import pulp
from collections import namedtuple

Point = namedtuple("Point", ['x', 'y'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


def tour_length(node_count, points, sol):
    # calculate the length of the tour
    obj = length(points[sol[-1]], points[sol[0]])
    for index in range(0, node_count - 1):
        obj += length(points[sol[index]], points[sol[index + 1]])

    return obj


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

    solution = ls_solution(points,node_count)
    #print(sorted(solution))
    obj = tour_length(node_count, points, solution)

    # prepare the solution in the specified output format
    output_data = str(obj) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data

def s_metropolis(t,N,s):
    n = random.randint(0,N-1)
    if f(n) <= f(s):
        return n
    else:
        if random.random()< exp(-(f(n)-f(s))/t):
            return n
        else:
            return s



def sa():
    s =generateInitialSolution()
    t = init_temp(s)
    s_min = s
    for k in range(1,max_search):
        s= ls_solution()
        if f(s) <= f(s_min):
            s_min = s
        t = update_temp(s,t)
    return s

def trivial_solution(points,node_count):
    # build a trivial solution
    # visit the nodes in the order they appear in the file
    return range(0, node_count)


def ls_solution(points,node_count):
    solution = naive_solution(points,node_count)
    current_value = tour_length(node_count, points, solution)
    iter_max = 100
    for i in range(0,iter_max):
        current_value, solution = try_swap(solution,node_count, points,current_value)
        if i%(iter_max/2)==0:
            print current_value
            #print solution
    return solution


def furthest_point( p_base, points):
    l_max = 0
    for i,p in enumerate(points):
        if length(points[0], p) > l_max:
            l_max = length(p_base, p)
            p_max = p
            i_max = i
    return i_max , p_max

def closest_point(p_base, points):
    l_min = 0
    for i,p in enumerate(points):
        if length(points[0], p) > l_min:
            l_min = length(p_base, p)
            p_min = p
            i_min = i
    return i_min , p_min


def rank_simple(vector):
    return sorted(range(len(vector)), key=vector.__getitem__)

def sort_closest_point(p_base, points):
    l_min = 0
    distance = [length(p_base, p) for p in points]
    return rank_simple(distance)


def is_on_the_way(p_in , p_out , p_c ,eps):
    v1 = Point(p_out.x - p_in.x,p_out.y - p_in.y)
    v2 = Point(p_c.x - p_in.x,p_c.y - p_in.y)
    return v1.x * v2.x + v1.y * v2.y + eps >= 0

def naive_solution(points,node_count):
    solution = []
    p_base = points[0]
    i_max , p_max = furthest_point(p_base, points)
    i2 , p2 = furthest_point(p_max, points)
    #print(str(i_max) + " "  +str(p_max))
    #print(str(i2) + " "  +str(p2))
    #print "-------------------"
    eps = 0.001
    visited = [False for p in points]
    old_sum = sum(visited)
    #print("old_sum " + str(old_sum))
    while not(all(visited)):
        for i in (sort_closest_point(p_base, points)):
            if not(visited[i]):
                if is_on_the_way(p_base,p_max,points[i],eps):
                    p_base = points[i]
                    solution.append(i)
                    visited[i] = True
                if i == i_max:
                    p_max = p2
                if i == i2 and p_max == p2:
                    #p_base = points[i]
                    #solution.append(i)
                    #visited[i] = True
                    p_max = points[0]
        #print("sum(visited) " + str(sum(visited)))
        if sum(visited) <= old_sum:
            eps = eps * 2
            #print(str(eps))
        old_sum = sum(visited)

    return solution

def try_swap(solution,node_count, points,current_value):
    c1 = random.randint(0,node_count-2)
    c2 = random.randint(c1+1,node_count-1)
    solution2 = swap(solution,c1,c2)
    new_value = tour_length(node_count, points, solution2)
    if new_value<current_value:
        print new_value
        return new_value,solution2
    return current_value, solution


def swap(solution, c1, c2):
    #print("node_count " + str(node_count) + " c1 " + str(c1) + " c2 " + str(c2))
    solution2=[]
    for i in solution:
        solution2.append(i)
    for i in range(c1, c2 + 1):
        j = c2 - i + c1
        #print("node_count " + str(node_count) + " i " + str(i) + " j " + str(j))
        solution2[j] = solution[i]
        #print(solution2)

    #print(solution)
    #print("sol2")
    return solution2

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

