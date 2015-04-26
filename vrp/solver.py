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
        for pos in range(0, len(customers) - 1):
            for c in range(0, len(customers) - 1):
                if T[v][pos][c].value() > 0.5:
                    vehicle_tours[v].append(customers[c + 1])
        print "v " + str(v) + " tour : " + str([c.index for c in vehicle_tours[v]])
    return vehicle_tours


def build_variables(vehicles, customers):
    res = []
    for v in vehicles:
        res.append([])
        for position in range(0, len(customers) - 1):
            res[v].append([])
            for chosen in range(0, len(customers) - 1):
                res[v][position].append(
                    pulp.LpVariable("x_v_" + str(v) + "_pos_" + str(position) + "_c_" + str(chosen + 1), 0, 1,
                                    'Binary'))
    return res

def build_int_variable(vehicle, size):
    return pulp.LpVariable("tour_size_" + str(vehicle), 0, size, 'Integer')


def to_value(T, vehicles, customers):
    t = []
    for v in vehicles:
        t.append([])
        for position in range(0, len(customers) - 1):
            t[v].append([])
            for chosen in range(0, len(customers) - 1):
                if T[v][position][chosen].value() > 0.5:
                    t[v][position].append(1)
                else:
                    t[v][position].append(0)
    return t

def pulp_solution(customers, vehicle_capacity, vehicle_count):
    print "capa " + str(vehicle_capacity)
    print "customers " + str([c.index for c in customers])
    print "demand " + str([c.demand for c in customers])
    depot = customers[0]
    model = pulp.LpProblem("Model", pulp.LpMinimize)
    vehicles = range(0, vehicle_count)
    customers_idx = range(0, len(customers) - 1)

    tours_size = []

    for v in vehicles:
        tours_size.append(build_int_variable(v, len(customers) - 1))


    # All tours serves all customer (except depot)
    model += sum(tours_size) == len(customers) - 1


    # T tour starts after depot and ends before depot
    T = build_variables(vehicles, customers)
    print(T)

    for v in vehicles:
        # tour_size customers served by each vehicle
        # print(tours_size[v])
        model += sum([sum(T[v][pos]) for pos in customers_idx]) == tours_size[v]
        for pos in range(1, len(customers) - 1):
            # One customer at most per position
            model += sum(T[v][pos - 1]) <= 1
            # Customers in first positions
            model += sum(T[v][pos - 1]) >= sum(T[v][pos])
        # less than capa
        model += sum([sum([customers[c + 1].demand * served for c, served in enumerate(T[v][pos])])
                      for pos in customers_idx]) <= vehicle_capacity

    model += sum([sum([sum([dist(customers[i + 1], customers[j + 1])
                            for i, ii in enumerate(T[v][pos]) if ii > 0.5
                            for j, jj in enumerate(T[v][pos + 1]) if jj > 0.5])
                       for pos in range(0, len(customers) - 2)]) for v in vehicles])
    + sum([sum([sum([dist(customers[i + 1], depot)
                     for i, ii in enumerate(T[v][pos]) if ii > 0.5
                     for j, jj in enumerate(T[v][pos + 1]) if jj == 0])
                for pos in range(0, len(customers) - 2)]) for v in vehicles])
    + sum([sum([dist(customers[i + 1], depot)
                for i, ii in enumerate(T[v][0]) if ii > 0.5])])

    model += sum([sum([sum([dist(customers[i + 1], customers[j + 1])
                            for i, ii in enumerate(T[v][pos]) if ii > 0
                            for j, jj in enumerate(T[v][pos + 1]) if jj > 0])
                       for pos in range(0, len(customers) - 2)]) for v in vehicles])
    + sum([sum([sum([dist(customers[i + 1], depot)
                     for i, ii in enumerate(T[v][pos]) if ii > 0
                     for j, jj in enumerate(T[v][pos + 1]) if jj == 0])
                for pos in range(0, len(customers) - 2)]) for v in vehicles])
    + sum([sum([dist(customers[i + 1], depot)
                for i, ii in enumerate(T[v][0]) if ii > 0.5])]) <= 100


    # All customers are served
    for c in customers_idx:
        model += sum(T[v][pos][c] for pos in customers_idx for v in vehicles) == 1

    model.solve(pulp.GLPK(msg=1))

    t = to_value(T, vehicles, customers)

    print t
    print str(sum([sum([sum([dist(customers[i + 1], customers[j + 1])
                             for i, ii in enumerate(t[v][pos]) if ii > 0.5
                             for j, jj in enumerate(t[v][pos + 1]) if jj > 0.5])
                        for pos in range(0, len(customers) - 2)]) for v in vehicles]))
    print str(sum([sum([sum([dist(customers[i + 1], depot)
                             for i, ii in enumerate(t[v][pos]) if ii > 0 and jj == 0
                             for j, jj in enumerate(t[v][pos + 1]) if jj == 0])
                        for pos in range(0, len(customers) - 2)]) for v in vehicles]))
    print str(sum([sum([dist(customers[i + 1], depot)
                        for i, ii in enumerate(t[v][0]) if ii > 0.5 for v in vehicles])]))

    print "objective=" + str(model.objective)

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

