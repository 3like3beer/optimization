#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
import pulp

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


def solution_cost(customers, facilities, solution, used):
    # calculate the cost of the solution
    obj = sum([f.setup_cost * used[f.index] for f in facilities])
    for customer in customers:
        obj += length(customer.location, facilities[solution[customer.index]].location)

    return obj


def create_use_list(facilities, solution):
    used = [0] * len(facilities)
    for facility_index in solution:
        used[facility_index] = 1
    return used


def trivial_solution(customers, facilities):
    # build a trivial solution
    # pack the facilities one by one until all the customers are served
    solution = [-1] * len(customers)
    capacity_remaining = [f.capacity for f in facilities]
    facility_index = 0
    for customer in customers:
        if capacity_remaining[facility_index] >= customer.demand:
            solution[customer.index] = facility_index
            capacity_remaining[facility_index] -= customer.demand
        else:
            facility_index += 1
            assert capacity_remaining[facility_index] >= customer.demand
            solution[customer.index] = facility_index
            capacity_remaining[facility_index] -= customer.demand
    used = create_use_list(facilities, solution)
    return solution, used

def pulp_solve(node_count,edges,opt):
    facility = pulp.LpProblem("Facility Model", pulp.LpMinimize)
    f_set = range(0,node_count)
    used =  [pulp.LpVariable("f" + str(f)  , 0,1, 'Binary') for f in f_set]
    # obj = pulp.LpVariable("objective",1,opt+1,'Integer')
    facility += sum([f.setup_cost * used[f.index] for f in f_set])
    # obj = sum([f.setup_cost * used[f.index] for f in facilities])
    # for customer in customers:
    #     obj += length(customer.location, facilities[solution[customer.index]].location)

    objective = pulp.LpAffineExpression(obj)
    facility.setObjective(objective)

    facility.solve(pulp.PULP_CBC_CMD(maxSeconds= 1000))

    out = []
    for f in f_set:
            if used[f].value()>0.5:
                #print ("col" + str(color) + "node" + str(node))
                out.append(f)
    #print out
    return out

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])
    
    facilities = []
    for i in range(1, facility_count+1):
        parts = lines[i].split()
        facilities.append(Facility(i-1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3])) ))

    customers = []
    for i in range(facility_count+1, facility_count+1+customer_count):
        parts = lines[i].split()
        customers.append(Customer(i-1-facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))

    solution, used = trivial_solution(customers, facilities)

    obj = solution_cost(customers, facilities, solution, used)

    # prepare the solution in the specified output format
    output_data = str(obj) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        input_data_file = open(file_location, 'r')
        input_data = ''.join(input_data_file.readlines())
        input_data_file.close()
        print 'Solving:', file_location
        print solve_it(input_data)
    else:
        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)'

