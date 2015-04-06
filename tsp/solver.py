#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import random
import math
from __builtin__ import enumerate
import pulp
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

    p = TSP(G, objective = 'time', start = 0,maxFunEvals=1500000,maxIter=50000) #, [optional] returnToStart={True}|False, constraints = ..., etc

    r = p.solve('sa') # also you can use some other solvers - sa, interalg, OpenOpt MILP solvers

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


# def s_metropolis(t,N,s):
#     n = random.randint(0,N-1)
#     if f(n) <= f(s):
#         return n
#     else:
#         if random.random()< exp(-(f(n)-f(s))/t):
#             return n
#         else:
#             return s


def update_temp(is_increase, t):
    epsilon = 0.01
    if is_increase:
        return t * (1 + epsilon)
    else:
        return max(t * 0.4,init_temp())


def sa_solution(points,node_count):
    taboo_s = set([])
    max_search = 3000
    s = naive_solution(points,node_count)
    min_val = tour_length(node_count, points, s)
    t = init_temp()
    s_min = s
    s_real_min = s
    real_min = min_val
    taboo_s.add(min_val)
    for k in range(1,max_search):
        current_value, s = try_swap(s_min,node_count, points)
        if current_value not in taboo_s:
            if current_value <= min_val:
                s_min = s
                min_val = current_value
                if current_value <= real_min:
                    s_real_min = s
                    real_min = current_value
                    print("Min " +str(current_value) )
                #taboo_s.add(min_val)
                #print(min_val)
                t = update_temp(False,t)
            else:
                if random.random()  < math.exp(-(current_value-min_val)/t):
                    #print("Update cur " + str(current_value) + " min " + str(min_val) + "prob" + str(math.exp(-(current_value-min_val)/t)))
                    s_min = s
                    min_val = current_value
                    #taboo_s.add(min_val)
                    #print(min_val)
                    t = update_temp(False,t)
                else:
                    #print("No update cur " + str(current_value) + " min " + str(min_val) + "prob" + str(math.exp(-(current_value-min_val)/t)))
                    t = update_temp(True,t)
        if k%(max_search/100)==0:
            print current_value
    s_real_min = ls_solution_given_init(node_count, points,s_real_min)
    return s_real_min

def init_temp():
    return 0.01


def trivial_solution(points,node_count):
    # build a trivial solution
    # visit the nodes in the order they appear in the file
    return range(0, node_count)


def ls_solution_given_init(node_count, points, solution):
    current_value = tour_length(node_count, points, solution)
    iter_max = 200
    for i in range(0, iter_max):
        new_value, solution2 = try_swap(solution, node_count, points)
        if new_value < current_value:
            # print new_value
            current_value, solution = new_value, solution2

        if i % (iter_max / 10) == 0:
            print current_value
            # print solution
    return solution


def ls_solution(points,node_count):
    solution = naive_solution(points,node_count)
    return ls_solution_given_init(node_count, points, solution)

def cx_solution(points,node_count):
    solution = convex_hull(points)
    point_set = {p:i for i,p in enumerate(points)}
    solution2 = [point_set[p] for p in solution]
    return solution2

def convex_hull(points):
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list.
    return lower[:-1] + upper[:-1]


# Example: convex hull of a 10-by-10 grid.
assert convex_hull([(i/10, i%10) for i in range(100)]) == [(0, 0), (9, 0), (9, 9), (0, 9)]

def get_rightmost(points):
    val, idx = max((p.x, idx) for (idx, p) in enumerate(points))
    return idx

def get_next_convex(point, points,angle):
    angles = [get_angle(points[point], p) - angle for p in points]
    #angles[point] = 2 * math.pi
    print angles
    print points
    l = None
    min_val, min_idx = min((val, idx) for (idx, val) in enumerate(angles))
    print ("min_val " + str(min_val))
    for (idx, val) in enumerate(angles):
        if val == min_val:
            if l is None:
                l = length(points[point],points[idx])
                out = idx
            else:
                if l > length(points[point],points[idx]):
                    l = length(points[point],points[idx])
                    out = idx
    return out,val

def get_angle(p1,p2):
    v = compute_vector(p1, p2)
    res = np.arctan2(v.y, v.x)
    if res <= 0:
        res = res + 2 * math.pi
    return res

def furthest_point( p_base, points):
    l_max = 0
    for i,p in enumerate(points):
        if length(p_base, p) > l_max:
            l_max = length(p_base, p)
            p_max = p
            i_max = i
    return i_max , p_max


def closest_point(p_base, points):
    l_min = 100000000000000
    for i,p in enumerate(points):
        if length(points[0], p) < l_min:
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


def compute_vector(p_in, p_out):
    return Point(p_out.x - p_in.x, p_out.y - p_in.y)


def is_on_the_way(p_in , p_out , p_c ,eps):
    v1 = compute_vector(p_in, p_out)
    v2 = compute_vector(p_in, p_c)
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


def swap_range(c1, c2,  node_count, solution):
    solution2 = [solution[i] for i in range(0, c1+1)]
    for i in range(c2, node_count):
        solution2.append(solution[i])
    for i in range(c1+1, c2):
        solution2.append(solution[i])
    return solution2


def try_swap(solution,node_count, points):
    a = random.random()
    if a < 0.5:
        c1 = random.randint(0,node_count-2)
        c2 = random.randint(c1+1,node_count-1)
        solution2 = swap_range(c1, c2, node_count, solution)
        #print("c1 " + str(c1) + " " + str(solution[c1]) +" c2 " +str(c2) +  " " + str(solution[c2])+  " " + str(solution[node_count-1]) )
        #print(solution)
        #print(solution2)
    else:
        c1 = random.randint(0,node_count-2)
        c2 = random.randint(c1+1,node_count-1)
        solution2 = swap(solution,c1,c2)
    new_value = tour_length(node_count, points, solution2)
    return new_value, solution2


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

