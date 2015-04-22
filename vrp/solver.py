#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple

import pulp


Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])


def dist(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x) ** 2 + (customer1.y - customer2.y) ** 2)


def trivial_solution(customers, vehicle_capacity, vehicle_count):
    # build a trivial solution
    # assign customers to vehicles starting by the largest customer demands
    depot = customers[0]
    vehicle_tours = []
    remaining_customers = set(customers)
    remaining_customers.remove(depot)
    for v in range(0, vehicle_count):
        # print "Start Vehicle: ",v
        vehicle_tours.append([])
        capacity_remaining = vehicle_capacity
        while sum([capacity_remaining >= customer.demand for customer in remaining_customers]) > 0:
            used = set()
            order = sorted(remaining_customers, key=lambda customer: -customer.demand)
            for customer in order:
                if capacity_remaining >= customer.demand:
                    capacity_remaining -= customer.demand
                    vehicle_tours[v].append(customer)
                    # print '   add', ci, capacity_remaining
                    used.add(customer)
            remaining_customers -= used

    return vehicle_tours


def cost(depot, vehicle_count, vehicle_tours):
    obj = 0
    for v in range(0, vehicle_count):
        vehicle_tour = vehicle_tours[v]
        if len(vehicle_tour) > 0:
            obj += dist(depot, vehicle_tour[0])
            for i in range(0, len(vehicle_tour) - 1):
                obj += dist(vehicle_tour[i], vehicle_tour[i + 1])
            obj += dist(vehicle_tour[-1], depot)

    return obj


def build_variable(node_in, node_out):
    if node_in > node_out:
        return pulp.LpVariable("x_in" + str(node_in) + "_out" + str(node_out), 0, 1, 'Binary')
    else:
        return 0.0


def pulp_solution(customers, vehicle_capacity, vehicle_count):
    depot = customers[0]
    vehicle_tours = []
    remaining_customers = set(customers)
    remaining_customers.remove(depot)
    model = pulp.LpProblem("Model", pulp.LpMinimize)
    vehicles = range(0, vehicle_count)
    T = []
    for v in vehicles:
        T.append([])
    model += sum(
        [dist(depot, T[i][0]) + sum([dist(j, k) + dist(T[i][len(T[i] - 1)]) for j in vehicles for k in vehicles]) for i
         in vehicles])

    for i in vehicles:
        model += sum(d[j] for j in T[i]) <= c

    model.solve()


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])

    customers = []
    for i in range(1, customer_count + 1):
        line = lines[i]
        parts = line.split()
        customers.append(Customer(i - 1, int(parts[0]), float(parts[1]), float(parts[2])))

    # the depot is always the first customer in the input
    depot = customers[0]

    vehicle_tours = trivial_solution(customers, vehicle_capacity, vehicle_count)

    # checks that the number of customers served is correct
    assert sum([len(v) for v in vehicle_tours]) == len(customers) - 1

    # calculate the cost of the solution; for each vehicle the length of the route
    obj = cost(depot, vehicle_count, vehicle_tours)

    # prepare the solution in the specified output format
    outputData = str(obj) + ' ' + str(0) + '\n'
    for v in range(0, vehicle_count):
        outputData += str(depot.index) + ' ' + ' '.join(
            [str(customer.index) for customer in vehicle_tours[v]]) + ' ' + str(depot.index) + '\n'

    return outputData


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

        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)'

