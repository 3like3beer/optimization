#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
import pulp

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacustomer', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])


def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


def cost(customer, facility):
    return length(customer.location, facility.location)


def solution_cost(customers, facilities, solution, used):
    # calculate the cost of the solution
    obj = sum([f.setup_cost * used[f.index] for f in facilities])
    for customer in customers:
        facility = facilities[solution[customer.index]]
        obj += cost(customer, facility)
    return obj


def create_use_list(facilities, solution):
    used = [0] * len(facilities)
    for facility_index in solution:
        used[facility_index] = 1
    return used

def get_unconnected(solution):
    return [i for i,c in enumerate(solution) if c == -1 ]

# At every moment, each customer j offers some money from its budget to each unopened facility i.
def get_offer(customer, facilities, facility, B, solution):
    offer2 = cost(customer,facility)
    if (solution[customer.index] == -1):
        # The amount of this offer is equal to max(Bj −cij,0) if j is unconnected,
        offer1 = B[customer.index]
    else:
        # and max(ci′j − cij,0) if it is connected to some other facility i′.
        offer1 = cost(customer , facilities[solution[customer.index]])
    return  max(offer1-offer2,0)


def open_facility(f, a, solution):
    print "Opening Facility " + str(f.index)
    for offer,c in a:
        if offer > 0:
            solution[c.index] = f.index
    print "New solution " + str(solution)


def open_if_possible(customers, facilities, f, B, solution):
    total_offer = 0
    offers = []
    for c in customers:
        offer = get_offer(c, facilities, f, B, solution)
        offers.append(offer)
        total_offer += offer
    if total_offer < f.setup_cost:
        return
    else:
        total_offer = 0
        a = zip(offers,customers)
        a.sort(key = lambda x:x[0], reverse = True)
        capacity = f.capacustomer
        assigned = []
        while total_offer < f.setup_cost:
            for offer,c in a:
                if capacity <= 0:
                    return
                else:
                    assigned.append(c)
                    capacity -= 1
                    total_offer += offer
                    #fin
        open_facility(f, a, solution)

def greedy_solution(customers, facilities):
    rate = 10000
    #1. At the beginning, all customers are unconnected,
    solution = [-1] * len(customers)
    # all facilities are unopened,
    # and the budget of every customer j, denoted by Bj, is initialized to 0.
    B = [0 for c in customers]

    #2. While there is an unconnected customer,
    unconnected = get_unconnected(solution)
    while unconnected:
        print "unconnected " + str(unconnected)
        used = create_use_list(facilities, solution)
        # increase the budget of each unconnected customer at the same rate,
        for uc in unconnected:
        # until one of the following events occurs:
            for f in facilities:
                if used[f.index] == 0:
                    # (a) For some unopened facility i, the total offer that it receives from customers is equal to the cost of opening i.
                    # In this case, we open facility i,
                    # and for every customer j (connected or unconnected) which has a non-zero offer to i, we connect j to i.
                    open_if_possible(customers, facilities, f, B, solution)
                else:
                    #(b) For some unconnected customer j,
                    # and some facility i that is already open,
                    # the budget of j is equal to the connection cost cij.
                    # In this case, we connect j to i.
                    if B[uc] >= cost(customers[uc] ,f):
                        solution[uc] = f.index
            B[uc] += rate
        unconnected = get_unconnected(solution)
        #print B
    used = create_use_list(facilities, solution)
    return solution, used


def trivial_solution(customers, facilities):
    # build a trivial solution
    # pack the facilities one by one until all the customers are served
    solution = [-1] * len(customers)
    capacustomer_remaining = [f.capacustomer for f in facilities]
    facility_index = 0
    for customer in customers:
        if capacustomer_remaining[facility_index] >= customer.demand:
            solution[customer.index] = facility_index
            capacustomer_remaining[facility_index] -= customer.demand
        else:
            facility_index += 1
            assert capacustomer_remaining[facility_index] >= customer.demand
            solution[customer.index] = facility_index
            capacustomer_remaining[facility_index] -= customer.demand
    used = create_use_list(facilities, solution)
    return solution, used

def pulp_solve(node_count,edges,opt):
    facility = pulp.LpProblem("Facility Model", pulp.LpMinimize)
    f_set = range(0,node_count)
    used =  [pulp.LpVariable("f" + str(f)  , 0,1, 'Binary') for f in f_set]
    # obj = pulp.LpVariable("objective",1,opt+1,'Integer')
    facility += sum([f.setup_cost * used[f.index] for f in f_set])
    obj = 1 #sum([f.setup_cost * used[f.index] for f in facilities])
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

    solution, used = greedy_solution(customers, facilities)

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

