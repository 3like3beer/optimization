#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
from random import shuffle
import math

import pulp


Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacustomer', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])


def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


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
    return [i for i, c in enumerate(solution) if c <= -0.5]


# At every moment, each customer j offers some money from its budget to each unopened facility i.
def get_offer(customer, facilities, facility, B, solution):
    offer2 = cost(customer, facility)
    if (solution[customer.index] == -1):
        # The amount of this offer is equal to max(Bj −cij,0) if j is unconnected,
        offer1 = B[customer.index]
    else:
        # and max(ci′j − cij,0) if it is connected to some other facility i′.
        offer1 = cost(customer, facilities[solution[customer.index]])
    return max(offer1 - offer2, 0)


def open_facility(f, assigned, solution, capacustomer_remaining):
    if assigned:
        for c in assigned:
            if solution[c.index] > -1:
                capacustomer_remaining[solution[c.index]] += c.demand
            solution[c.index] = f.index
            capacustomer_remaining[f.index] -= c.demand
            #print "Opening Facility " + str(f.index) + " assigning " + to_string(
            #   [c.index for c in assigned]) + " To assign " + to_string(get_unconnected(solution))


def to_string(list):
    if len(list) > 10:
        return "list of size: " + str(len(list))
    else:
        return str(list)


def open_if_possible(customers, facilities, f, B, solution, capacustomer_remaining):
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
        a = zip(offers, customers)
        a.sort(key=lambda x: x[0], reverse=True)
        capacity = f.capacustomer
        assigned = []
        while total_offer < f.setup_cost and capacity >= c.demand:
            for offer, c in a:
                if capacity >= c.demand:
                    assigned.append(c)
                    capacity -= c.demand
                    total_offer += offer
        open_facility(f, assigned, solution, capacustomer_remaining)


def get_demand(customers, f, solution):
    demand = 0
    for c in customers:
        if solution[c.index] == f.index:
            demand += c.demand
    return demand


def test_swap(open_f, closed_f, customers, facilities, solution):
    new_solution = [s for s in solution]
    cur_cost = solution_cost(customers, facilities, solution, create_use_list(facilities, solution))
    demand = get_demand(customers, open_f, solution)
    if demand < closed_f.capacustomer:
        # print "test swaping " + str(open_f.index) + " and " + str(closed_f.index)
        for c in customers:
            if new_solution[c.index] == open_f.index:
                new_solution[c.index] = closed_f.index
        assert create_use_list(facilities, new_solution)[open_f.index] == 0
        #assert create_use_list(facilities, new_solution)[closed_f.index] == 1
    if solution_cost(customers, facilities, new_solution, create_use_list(facilities, new_solution)) < cur_cost:
        # print "new cost " + str(solution_cost(customers, facilities, new_solution, create_use_list(facilities, new_solution)))
        return new_solution, create_use_list(facilities, new_solution)
    else:
        return solution, create_use_list(facilities, solution)


def opt_local(customers, facilities, min_cost, min_solution, min_used):
    old_cost = 10000000000000000
    while min_cost < old_cost:
        old_cost = min_cost
        for open_f in [f for f in facilities if min_used[f.index] > 0.5]:
            for closed_f in [f for f in facilities if min_used[f.index] < 0.5]:
                solution, used = test_swap(open_f, closed_f, customers, facilities, min_solution)
                for c in customers:
                    for old_f in [f for f in facilities if min_used[f.index] > 0.5]:
                        for new_f in [f for f in facilities if min_used[f.index] > 0.5]:
                            if solution[c.index] == old_f and new_f.capacustomer > c.demand + get_demand(customers, f,
                                                                                                         solution):
                                new_sol = [s for s in solution]
                                new_sol[c.index] = new_f
                                if solution_cost(customers, facilities, new_sol,
                                                 create_use_list(facilities, new_sol)) < cur_cost:
                                    solution = new_sol
                                    used = create_use_list(facilities, new_sol)
                cur_cost = solution_cost(customers, facilities, solution, used)
                if cur_cost < min_cost:
                    min_cost = cur_cost
                    min_solution = solution
                    min_used = used
    return min_solution, min_used


def local_greedy2(customers, facilities):
    min_solution, min_used = local_greedy(customers, facilities)
    min_cost = solution_cost(customers, facilities, min_solution, min_used)
    print "shuffle_greedy cost " + str(min_cost)
    min_solution, min_used = opt_local(customers, facilities, min_cost, min_solution, min_used)
    return min_solution, min_used


def local_greedy(customers, facilities):
    nb = 5
    i = 0
    min_cost = 10000000000000000
    while i < nb:

        if nb>10 and i % (nb / 10) == 0:
            print "nb " + str(i) + "/ " + str(nb)
        solution, used = greedy_solution(customers, facilities)
        cur_cost = solution_cost(customers, facilities, solution, used)
        if cur_cost < min_cost:
            min_cost = cur_cost
            min_solution = solution
            min_used = used
            print str(min_cost)
        i += 1
    return min_solution, min_used


def greedy_solution(customers, facilities):
    rate = 100000
    # 1. At the beginning, all customers are unconnected,
    solution = [-1] * len(customers)
    # all facilities are unopened,
    # and the budget of every customer j, denoted by Bj, is initialized to 0.
    B = [0 for c in customers]

    capacustomer_remaining = [f.capacustomer for f in facilities]
    shuffle_facilities = [f for f in facilities]
    shuffle(shuffle_facilities)

    unconnected = get_unconnected(solution)
    shuffle_unconnected = [c for c in unconnected]
    shuffle(shuffle_unconnected)

    # 2. While there is an unconnected customer,
    unconnected = get_unconnected(solution)
    while unconnected:
        used = create_use_list(facilities, solution)
        # increase the budget of each unconnected customer at the same rate,
        while unconnected:
            for uc in unconnected:
                B[uc] += rate
                # until one of the following events occurs:
                for f in shuffle_facilities:
                    used = create_use_list(facilities, solution)
                    if used[f.index] < 1:
                        # (a) For some unopened facility i, the total offer that it receives from customers is equal to the cost of opening i.
                        # In this case, we open facility i,
                        # and for every customer j (connected or unconnected) which has a non-zero offer to i, we connect j to i.
                        open_if_possible(customers, facilities, f, B, solution, capacustomer_remaining)
                        unconnected = get_unconnected(solution)
                        if not unconnected:
                            used = create_use_list(facilities, solution)
                            return opt_local(customers, facilities, solution_cost(customers, facilities, solution, used) , solution, used)
                            #print ("unconnected 1 " + to_string(unconnected))
                    else:
                        unconnected = get_unconnected(solution)
                        # print ("unconnected 2 " + to_string(unconnected))
                        for uc in unconnected:
                            #(b) For some unconnected customer j,
                            # and some facility i that is already open,
                            # the budget of j is equal to the connection cost cij.
                            # In this case, we connect j to i.
                            if solution[uc] < -0.5:
                                if B[uc] >= cost(customers[uc], f): #and used[f.index] > 0.5:
                                    if capacustomer_remaining[f.index] > customers[uc].demand:
                                        #print "connect customer " + str(uc) + " to facility " + str(f.index)
                                        #print "Current solution " + str(solution)
                                        solution[uc] = f.index
                                        capacustomer_remaining[f.index] -= customers[uc].demand
                        if not unconnected:
                            used = create_use_list(facilities, solution)
                            return opt_local(customers, facilities, solution_cost(customers, facilities, solution, used) , solution, used)
            unconnected = get_unconnected(solution)
            # print ("unconnected " + to_string(unconnected))
            #print B
    used = create_use_list(facilities, solution)
    return opt_local(customers, facilities, solution_cost(customers, facilities, solution, used) , solution, used)


def trivial_solution(customers, facilities):
    # build a trivial solution
    # pack the facilities one by one until all the customers are served
    solution = [-1] * len(customers)
    capacustomer_remaining = [f.capacustomer for f in facilities]
    facilities_index = [f.index for f in facilities]
    shuffle(facilities_index)
    i = 0
    facility_index = facilities_index[i]
    for customer in customers:
        if capacustomer_remaining[facility_index] >= customer.demand:
            solution[customer.index] = facility_index
            capacustomer_remaining[facility_index] -= customer.demand
        else:
            i += 1
            facility_index = facilities_index[i]
            assert capacustomer_remaining[facility_index] >= customer.demand
            solution[customer.index] = facility_index
            capacustomer_remaining[facility_index] -= customer.demand
    used = create_use_list(facilities, solution)
    return solution, used


def pulp_solve(node_count, edges, opt):
    facility = pulp.LpProblem("Facility Model", pulp.LpMinimize)
    f_set = range(0, node_count)
    used = [pulp.LpVariable("f" + str(f), 0, 1, 'Binary') for f in f_set]
    # obj = pulp.LpVariable("objective",1,opt+1,'Integer')
    facility += sum([f.setup_cost * used[f.index] for f in f_set])
    obj = 1  # sum([f.setup_cost * used[f.index] for f in facilities])
    # for customer in customers:
    # obj += length(customer.location, facilities[solution[customer.index]].location)

    objective = pulp.LpAffineExpression(obj)
    facility.setObjective(objective)

    facility.solve(pulp.PULP_CBC_CMD(maxSeconds=1000))

    out = []
    for f in f_set:
        if used[f].value() > 0.5:
            # print ("col" + str(color) + "node" + str(node))
            out.append(f)
    # print out
    return out


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])

    facilities = []
    for i in range(1, facility_count + 1):
        parts = lines[i].split()
        facilities.append(Facility(i - 1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3]))))

    customers = []
    for i in range(facility_count + 1, facility_count + 1 + customer_count):
        parts = lines[i].split()
        customers.append(Customer(i - 1 - facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))

    solution, used = local_greedy2(customers, facilities)

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

