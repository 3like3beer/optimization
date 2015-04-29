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


def tour_cost(depot, vehicle_tour):
    tour_obj = 0
    if len(vehicle_tour) > 0:
        tour_obj += dist(depot, vehicle_tour[0])
        print "obj for tour from depot : " + str(tour_obj)
        for i in range(0, len(vehicle_tour) - 1):
            tour_obj += dist(vehicle_tour[i], vehicle_tour[i + 1])
            print "obj for tour from between : " + str(tour_obj)
        tour_obj += dist(vehicle_tour[-1], depot)
    print "Tot obj for tour : " + str(tour_obj)
    return tour_obj


def cost(depot, vehicle_count, vehicle_tours):
    obj = 0
    for v in range(0, vehicle_count):
        vehicle_tour = vehicle_tours[v]
        obj += tour_cost(depot, vehicle_tour)

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


def build_tours(t, vehicle_count):
    vehicle_tours = []
    for v in range(0, vehicle_count):
        vehicle_tours.append([])
        y = 0
        first = True
        while y > 0 or first:
            first = False
            x = y
            y = next(i for i, y in enumerate(t[v][x]) if y > 0)
            print "v " + str(v) + " y " + str(y)
            if y > 0:
                vehicle_tours[v].append(y)
    return vehicle_tours


def build_variables(vehicles, customers):
    res = []
    for v in vehicles:
        res.append([])
        for from_c in range(0, len(customers)):
            res[v].append([])
            for to_c in range(0, len(customers)):
                res[v][from_c].append(
                    pulp.LpVariable("x_v_" + str(v) + "_from_" + str(from_c) + "_to_" + str(to_c), 0, 1,
                                    'Binary'))
    return res

def build_int_variable(vehicle, size):
    return pulp.LpVariable("tour_size_" + str(vehicle), 0, size, 'Integer')


def to_value(T, vehicles, customers):
    t = []
    for v in vehicles:
        t.append([])
        for from_c in range(0, len(customers)):
            t[v].append([])
            for to_c in range(0, len(customers)):
                if T[v][from_c][to_c].value() > 0.5:
                    t[v][from_c].append(1)
                else:
                    t[v][from_c].append(0)
    return t

def pulp_solution(customers, vehicle_capacity, vehicle_count):
    print "capa " + str(vehicle_capacity)
    print "customers " + str([to_c.index for to_c in customers])
    print "demand " + str([to_c.demand for to_c in customers])
    depot = customers[0]
    model = pulp.LpProblem("Model", pulp.LpMinimize)
    vehicles = range(0, vehicle_count)
    customers_idx = range(0, len(customers))




    # T tour starts after depot and ends before depot
    T = build_variables(vehicles, customers)
    # customer visited by v
    z = [[pulp.LpVariable("x_v_" + str(v) + "_c_" + str(c), 0, 1, 'Binary') for c in range(0, len(customers))] for v in
         vehicles]
    # print(T)

    model += sum([sum(
        [sum([T[v][from_c][to_c] * dist(customers[from_c], customers[to_c]) for to_c in customers_idx]) for from_c
         in customers_idx]
    ) for v in vehicles])

    for to_c in customers_idx:
        model += sum(z[v][to_c] for v in vehicles) == 1
    #
    # for from_c in customers_idx:
    # if from_c > 0:
    #         model += sum(T[v][from_c][to_c] for to_c in customers_idx for v in vehicles) == z[from_c]



    for v in vehicles:
        # Return depot
        # model += sum(T[v][from_c][0] for from_c in customers_idx) == 1
        # Leaves depot
        model += sum([T[v][0][to_c] for to_c in customers_idx]) == z[v][0]

        # for to_c in range(1, len(customers) - 1):
        # One customer at most per from_c
        # model += sum(T[v][to_c]) <= 1
        # model += T[v][to_c][to_c] == 0

        # Less than capa
        model += sum([sum([customers[to_c].demand * z[v][to_c] for to_c in customers_idx])
                      for from_c in customers_idx]) <= vehicle_capacity * z[v][0]
        # Flux loops (nb in == nb out)
        for c in range(1, len(customers)):
            # model += sum(T[v][c][to_c] for to_c in customers_idx) == sum(T[v][from_c][c] for from_c in customers_idx)
            model += sum([T[v][c][to_c] for to_c in customers_idx]) == z[v][c]
            model += sum([T[v][from_c][c] for from_c in customers_idx]) == z[v][c]

    model.solve(pulp.PULP_CBC_CMD(msg=0, maxSeconds=100))
    print(model)
    t = to_value(T, vehicles, customers)

    for vehicle in vehicles:
        # print "tour of " + str(vehicle)
        print(t[vehicle])
        print "cost of vehicle " + str(vehicle)
        cost = sum([sum(
            [sum([t[vehicle][from_c][to_c] * dist(customers[from_c], customers[to_c]) for to_c in customers_idx]) for
             from_c in customers_idx])]) + t[v][from_c][to_c] * dist(depot, customers[len(customers) - 2])
        print [[(from_c, to_c, customers[from_c].index, customers[to_c].index,
                 t[vehicle][from_c][to_c] * dist(customers[from_c], customers[to_c])) for to_c in customers_idx if
                t[vehicle][from_c][to_c] > 0] for
               from_c in customers_idx]
        print cost
    #print "objective=" + str(model.objective)

    print vehicle_capacity
    print "computed capa"
    for v in vehicles:
        print t[v]
        print [z[v][c].value() for c in range(0, len(customers))]
        print (sum([sum([customers[to_c].demand * t[v][from_c][to_c] for to_c in customers_idx])
                    for from_c in customers_idx]))
    vehicle_tours = build_tours(t, vehicle_count)
    print vehicle_tours
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

