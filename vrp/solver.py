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


def build_variable(c_in, c_out, v):
    return pulp.LpVariable("x_veh" + str(v) + "_in" + str(c_in) + "_out" + str(c_out), 0, 1, 'Binary')


def list_served(Ti):
    served = []
    for k, i in enumerate(Ti):
        if k == 0:
            served.append(k)
        else:
            if i == 1:
                served.append(k)
    return served


def build_tours(T, customers, vehicle_count):
    vehicle_tours = []
    for v in range(0, vehicle_count):
        vehicle_tours.append([])
        for c in range(1, len(customers)):
            if T[v][c].value() > 0.5:
                vehicle_tours[v].append(customers[c])
        print "v " + str(v) + " tour : " + str([c.index for c in vehicle_tours[v]])
    return vehicle_tours


def build_variables(vehicles, customers):
    for v in vehicles:
        for position in range(0, len(customers)):
            for chosen in range(0, len(customers)):
                if chosen == 0:
                    return 1
                else:
                    return pulp.LpVariable("x_v_" + str(v) + "_pos_" + str(position) + "_c_" + str(chosen), 0, 1,
                                           'Binary')


def build_int_variable(vehicle, size):
    return pulp.LpVariable("tour_size_" + str(vehicle), 0, size, 'Integer')


def pulp_solution(customers, vehicle_capacity, vehicle_count):
    print "customers " + str([c.index for c in customers])
    print "demand " + str([c.demand for c in customers])
    depot = customers[0]
    vehicle_tours = []
    remaining_customers = set(customers)
    remaining_customers.remove(depot)
    model = pulp.LpProblem("Model", pulp.LpMinimize)
    vehicles = range(0, vehicle_count)
    cs = range(0, len(customers))

    tours_size = []

    for v in vehicles:
        tours_size.append(build_int_variable(v, len(customers) - 1))


    # All tours serves all customer
    model += sum(tours_size) == len(customers) - 1


    #T tour starts at depot ends before 0
    T = build_variables(vehicles, customers)

    for v in vehicles:
        #tour_size customers served by each vehicle
        model += sum([sum(T[v][pos]) for pos in range(1, len(customers))]) == tours_size[v]
        for pos in range(1, len(customers)):
            #One customer for the 1st positions then 0 (no gap)
            model += sum(T[v][pos - 1]) >= sum(T[v][pos])

    out_edges = [[[build_variable(c_in, c_out, v) for c_in in cs] for c_out in cs] for v in vehicles]

    model += sum([sum([sum([dist(customers[i], customers[j])
                            for i in T[v][pos] if T[v][pos][i] > 0
                            for j in T[v][pos] if T[v][pos + 1][j] > 0]) for pos in T[v]]) for v in vehicles])
    + sum([dist(customers[tours_size[v]], depot) for v in vehicles])


    model += sum(
        sum(
            [sum([out_edges[v][c_in][c_out] * dist(customers[c_in], customers[c_out]) for c_in in cs])
             for c_out in cs])
        for v in vehicles)

    for i in vehicles:
        model += sum(customers[c].demand * is_served for c, is_served in enumerate((T[i]))) <= vehicle_capacity

    for j in range(1, len(customers)):
        model += sum(T[i][j] for i in vehicles) == 1

    model.solve()

    vehicle_tours = build_tours(T, customers, vehicle_count)
    print vehicle_capacity
    print ([sum(j.demand for j in t) for t in vehicle_tours])

    return vehicle_tours


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

    vehicle_tours = pulp_solution(customers, vehicle_capacity, vehicle_count)
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

