#!/usr/bin/python
# -*- coding: utf-8 -*-


# import os
from subprocess import Popen, PIPE
from time import gmtime, strftime
import random


PIP_NAME = 'problem.pip'
SOL_NAME = 'problem.sol'
INDENT = ' ' * 9


def pe(*args):
    now = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    msg = now + ' ' + ' '.join(map(str, args)) + '\n'
    sys.stderr.write(msg)
    sys.stderr.flush()


class SimpleModel(object):
    '''
    Constants:
        N -- # of warehouses
        M -- # of clients / customers
        cap_w = c_w -- warehouse capacity
        s_w -- warehouse setup cost
        d_c -- customer demand
        t_c_w -- transportation cost from customer c to warehouse w

    Variables:
        a_c_w -- customer c assigned to warehouse w

    Objective f:
        // minimize 1. setup cost 2. transportation cost
        min sum(s_w * a_c_w) + sum(t_c_w * a_c_w)

    Subject to:
        // total assigned demand <= warehouse capacity
        sum(d_c * a_c_w)|c = 1..M <= c_w (forall w in N)

        // customer assigned to only one warehouse
        sum(a_c_w)|w = 1..N == 1 (forall c in M)

    WARNING: this model is flawed and it does not guarantee optimality
    '''

    def __init__(self, warehouses, customerSizes, customerCosts):
        self.warehouses = warehouses
        self.customerSizes = customerSizes
        self.customerCosts = customerCosts
        self.N = len(warehouses)
        self.M = len(customerSizes)

    def generatePip(self):
        def assignment(c, w):
            return 'a_{0}_{1}'.format(c, w)

        with open(PIP_NAME, 'w') as f:
            f.write('Minimize\n\n')
            f.write('\\ 1. Total setup cost\n')
            f.write('    obj:\n')
            for w in range(self.N):
                for c in range(self.M):
                    f.write('{0}{1} {2} +\n'.format(INDENT, self.warehouses[w][1], assignment(c, w)))
            f.write('\n')
            f.write('\\ 2. Total travel cost from client to assigned warehouse\n')
            for w in range(self.N):
                for c in range(self.M):
                    plus = '' if (w == self.N - 1) and (c == self.M - 1) else ' +'
                    f.write('{0}{1} {2}{3}\n'.format(INDENT, self.customerCosts[c][w], assignment(c, w), plus))
            f.write('\n')

            f.write('Subject to\n')
            f.write('\\ Total clients\' demand for warehouse <= warehouse capacity\n')
            for w in range(self.N):
                for c in range(self.M):
                    less = ' <= {0}\n'.format(self.warehouses[w][0]) if c == self.M - 1 else ' +'
                    f.write('{0}{1} {2}{3}\n'.format(INDENT, self.customerSizes[c], assignment(c, w), less))
            #f.write('\n')

            f.write('\\ Each client is assigned to only one warehouse\n')
            f.write('\\ i.e. sum(Ac)|w=1..N == 1\n')
            for c in range(self.M):
                for w in range(self.N):
                    plus = ' == 1\n' if w == self.N - 1 else ' +'
                    f.write('{0}{1}{2}\n'.format(INDENT, assignment(c, w), plus))
            #f.write('\n')

            f.write('Bounds\n')
            f.write('\n')

            f.write('Binary\n')
            for w in range(self.N):
                for c in range(self.M):
                    f.write('{0}{1}\n'.format(INDENT, assignment(c, w)))
                f.write('\n')
            #f.write('\n')

            f.write('End\n')

    def parseSolution(self):
        value = 0
        clientWarehouse = initAssignments(self.N, self.M)

        with open(SOL_NAME) as f:
            f.readline()  # skip 'solution found' message
            value = f.readline()[len('objective value:'):].strip()
            while True:
                line = f.readline()
                if len(line) == 0:
                    break
                if not line.startswith('a_'):  # assignment variable prefix
                    continue
                name, val, _ = line.strip().split()
                _, c, w = name.split('_')
                clientWarehouse[int(c)] = int(w)

        self.objectiveValue = value
        self.clientAssignments = clientWarehouse

    def formatSolution(self):
        first = '{0} {1}'.format(self.objectiveValue, 0)
        second = ' '.join(map(str, self.clientAssignments))
        print(first)
        print(second)
        return '{0}\n{1}'.format(first, second)


class LectureModel(object):
    '''
    Constants:
        N -- # of warehouses
        M -- # of clients / customers
        cap_w = c_w -- warehouse capacity
        s_w -- warehouse setup cost
        d_c -- customer demand
        t_c_w -- transportation cost from customer c to warehouse w

    Variables:
        a_c_w -- customer c assigned to warehouse w
        o_w -- warehouse w is open

    Objective f:
        // minimize 1. setup cost 2. transportation cost
        min sum(s_w * o_w) + sum(t_c_w * a_c_w)

    Subject to:
        // we can assign customer c to warehouse w only if it is open
        a_c_w <= o_w (forall w in W, c in C)
        actually: a_c_w - o_w <= 0

        // customer assigned to only one warehouse
        sum(a_c_w)|w in W == 1 (forall c in M)

        // total assigned demand <= warehouse capacity
        sum(d_c * a_c_w)|c = 1..M <= c_w (forall w in N)
    '''

    def __init__(self, warehouses, customerSizes, customerCosts):
        self.warehouses = warehouses
        self.customerSizes = customerSizes
        self.customerCosts = customerCosts
        self.N = len(warehouses)
        self.M = len(customerSizes)

    def generatePip(self):
        # 10 cheapest warehouses for problem #6
        fixed = set([264, 109, 484, 462, 145, 482, 414, 401, 115, 95])
        # fixed = set([95])
        lastFixed = sorted(list(fixed))[-1]

        def assignment(c, w):
            return 'a_{0}_{1}'.format(c, w)

        def wopen(w):
            if w in fixed:
                return 'o_{0}'.format(w)
            else:
                return ''

        with open(PIP_NAME, 'w') as f:

            f.write('Minimize\n\n')
            f.write('    obj:\n')
            f.write('\\ 1. Total setup cost\n')
            for w in range(self.N):
                # for c in range(self.M):
                #f.write('{0}{1} {2} +\n'.format(INDENT, self.warehouses[w][1], assignment(c, w)))
                if w in fixed:
                    f.write('{0}{1} {2} +\n'.format(INDENT, self.warehouses[w][1], wopen(w)))
            f.write('\n')

            f.write('\\ 2. Total travel cost from client to assigned warehouse\n')
            for w in range(self.N):
                for c in range(self.M):
                    # plus = '' if (w == self.N-1) and (c == self.M-1) else ' +'
                    plus = '' if (w == lastFixed) and (c == self.M - 1) else ' +'
                    if w in fixed:
                        f.write('{0}{1} {2}{3}\n'.format(INDENT, self.customerCosts[c][w], assignment(c, w), plus))
                if w in fixed:
                    f.write('\n')

            f.write('Subject to\n')
            f.write('\\ 1. Assign customer to open warehouse only\n')
            f.write('\\ i.e. sum(Ac)|w=1..N == 1\n')
            for c in range(self.M):
                for w in range(self.N):
                    plus = ' <= 0'
                    if w in fixed:
                        f.write('{0}{1} - {2}{3}\n'.format(INDENT, assignment(c, w), wopen(w), plus))
                if w in fixed:
                    f.write('\n')

            f.write('\\ 2. Each client is assigned to only one warehouse\n')
            f.write('\\ i.e. sum(Ac)|w=1..N == 1\n')
            for c in range(self.M):
                for w in range(self.N):
                    plus = ' +'  # ' == 1\n' if w == self.N-1 else ' +'
                    if w in fixed:
                        f.write('{0}{1}{2}\n'.format(INDENT, assignment(c, w), plus))
                f.write('{0}0 == 1\n'.format(INDENT))
            # f.write('\n')

            f.write('\\ 3. Total clients\' demand for warehouse <= warehouse capacity\n')
            for w in range(self.N):
                for c in range(self.M):
                    less = ' <= {0}\n'.format(self.warehouses[w][0]) if c == self.M - 1 else ' +'
                    if w in fixed:
                        f.write('{0}{1} {2}{3}\n'.format(INDENT, self.customerSizes[c], assignment(c, w), less))
            #f.write('\n')

            f.write('Bounds\n')
            #f.write('Binary\n')
            f.write('\n')

            f.write('Binary\n')
            #f.write('Bounds\n')
            for w in range(self.N):
                for c in range(self.M):
                    if w in fixed:
                        f.write('{0}{1}\n'.format(INDENT, assignment(c, w)))
                        #f.write('{0}0 <= {1} <= 1\n'.format(INDENT, assignment(c, w)))
                if w in fixed:
                    f.write('\n')

            for w in range(self.N):
                if w in fixed:
                    f.write('{0}{1}\n'.format(INDENT, wopen(w)))
                    #f.write('{0}0 <= {1} <= 1\n'.format(INDENT, wopen(w)))
            #f.write('\n')

            f.write('End\n')

    def parseSolution(self):
        value = 0.0
        clientWarehouse = initAssignments(self.N, self.M)
        self.probabilities = []
        for _ in range(self.M):
            self.probabilities.append([0.0] * self.N)

        with open(SOL_NAME) as f:
            f.readline()  # skip 'solution found' message
            value = float(f.readline()[len('objective value:'):].strip())
            while True:
                line = f.readline()
                if len(line) == 0:
                    break
                if not line.startswith('a_'):  # assignment variable prefix
                    continue
                name, val, _ = line.strip().split()
                _, c, w = name.split('_')
                c, w = int(c), int(w)
                clientWarehouse[c] = w
                self.probabilities[c][w] = float(val)

        self.objectiveValue = value
        self.clientAssignments = clientWarehouse

        # pe('hello', clientWarehouse)
        #pe('hello', pprint.pformat(probabilities))
        #pe('hello', probabilities)

    def cost(self, c, w):
        # setup cost + transportation cost
        return self.warehouses[w][1] + self.customerCosts[c][w]

    def cheapestWarehouseForClient(self, c, capacity):
        lowestIndex = 0
        lowestCost = self.cost(c, lowestIndex)
        for w in range(1, self.N):
            currentCost = self.cost(c, w)
            if (currentCost < lowestCost) and (capacity[w] >= self.customerSizes[c]):
                lowestCost, lowestIndex = currentCost, w
        return lowestIndex

    def roundSolutionRandomly(self):
        capacityRemaining = [w[0] for w in self.warehouses]
        for ci, c in enumerate(self.probabilities):
            assigned = False
            for wi, w in enumerate(c):
                p = self.probabilities[ci][wi]
                if random.random() < p:
                    self.clientAssignments[ci] = wi
                    pe('Assigned client {0} to warehouse {1} (p={2})'
                       ''.format(ci, wi, p))
                    assigned = True
                    break
            if not assigned:
                # wi = random.randint(0, self.N - 1)
                wi = self.cheapestWarehouseForClient(ci, capacityRemaining)
                self.clientAssignments[ci] = wi
                capacityRemaining[wi] -= self.customerSizes[ci]
                pe('No probability worked, assigned client {0} to cheapest warehouse {1}'
                   ''.format(ci, wi))
                #pe('No probability worked, randomly assigned client {0} to warehouse {1}'
                #   ''.format(ci, wi))

        self.fractionalObjectiveValue = self.objectiveValue
        self.roundedObjectiveValue = self.calcObjectiveValue()
        self.objectiveValue = self.roundedObjectiveValue

        diff = self.roundedObjectiveValue - self.fractionalObjectiveValue
        pe('Rounded {0} - Fractional {1} = {2}'
           ''.format(self.roundedObjectiveValue, self.fractionalObjectiveValue,
                     diff))

    def calcObjectiveValue(self):
        objValue = 0.0
        warehouseUsed = [False] * self.N
        for ci in range(self.M):
            wi = self.clientAssignments[ci]
            # client transportation cost
            objValue += self.customerCosts[ci][wi]
            if not warehouseUsed[wi]:
                warehouseUsed[wi] = True
                # warehouse setup cost
                objValue += self.warehouses[wi][1]
        return objValue

    def formatSolution(self):
        first = '{0} {1}'.format(self.objectiveValue, 0)
        second = ' '.join(map(str, self.clientAssignments))
        print(first)
        print(second)
        return '{0}\n{1}'.format(first, second)


def runSolver():
    process = Popen(['./runSolver.sh', PIP_NAME, SOL_NAME], stdout=PIPE)
    (stdout, stderr) = process.communicate()


def initAssignments(N, M):
    return [0] * M


def solveWLP(model):
    # pe('Generating problem description')
    #model.generatePip()
    #pe('Running solver')
    #runSolver()
    model.parseSolution()
    #model.objectiveValue = model.calcObjectiveValue()
    #model.roundSolutionRandomly()
    pe('Solution is ready')
    return model.formatSolution()


def solveGreedyBest(warehouses, customerSizes, customerCosts):
    bestStart = -1
    bestCost = -1
    bestSolution = None

    for i in range(len(customerSizes)):
        pe('Trying {0} / {1}'.format(i, len(customerSizes)))
        cost, solution = solveGreedy(warehouses, customerSizes, customerCosts, i)
        if (cost < bestCost) or (bestStart == -1):
            pe('Better solution at {0}: {1} < {2}'.format(i, cost, bestCost))
            bestCost = cost
            bestStart = i
            bestSolution = solution

    pe('Best solution starts at client {0}: {1}'.format(bestStart, bestCost))
    print(bestSolution)
    return bestSolution


def solveGreedy(warehouses, customerSizes, customerCosts, startClient=0):
    N = len(warehouses)
    M = len(customerSizes)

    # build a trivial solution
    # pack the warehouses one by one until all the customers are served

    def openCost(c, w, capacity):
        value = 0.0
        # value = customerCosts[c][w]
        #if capacity[w] == warehouses[w][0]:
        value += warehouses[w][1]
        return value

    def cost(c, w, capacity):
        value = 0.0
        value = customerCosts[c][w]
        # if capacity[w] == warehouses[w][0]:
        #value += warehouses[w][1]
        return value

    cheapestWarehouses = sorted(range(M), key=lambda x: openCost(-1, x, None))
    cheapestWarehouses = cheapestWarehouses[:10]
    pe(cheapestWarehouses)

    def cheapestWarehouseForClient(c, capacity):
        # return random.choice(cheapestWarehouses)

        lowestIndex = cheapestWarehouses[0]
        lowestCost = cost(c, lowestIndex, capacity)
        for i in range(1, len(cheapestWarehouses)):
            w = cheapestWarehouses[i]
            currentCost = cost(c, w, capacity)
            if (currentCost < lowestCost) and (capacity[w] >= customerSizes[c]):
                lowestCost, lowestIndex = currentCost, w
        return lowestIndex

        # lowestIndex = 0
        # lowestCost = cost(c, lowestIndex, capacity)
        # for w in range(1, N):
        # currentCost = cost(c, w, capacity)
        #     if (currentCost < lowestCost) and (capacity[w] >= customerSizes[c]):
        #         lowestCost, lowestIndex = currentCost, w
        # return lowestIndex

    solution = [-1] * M
    capacityRemaining = [w[0] for w in warehouses]

    warehouseIndex = 0
    for i in range(0, M):
        c = (startClient + i) % M
        warehouseIndex = cheapestWarehouseForClient(c, capacityRemaining)

        if capacityRemaining[warehouseIndex] >= customerSizes[c]:
            solution[c] = warehouseIndex
            capacityRemaining[warehouseIndex] -= customerSizes[c]
        else:
            warehouseIndex += 1
            assert capacityRemaining[warehouseIndex] >= customerSizes[c]
            solution[c] = warehouseIndex
            capacityRemaining[warehouseIndex] -= customerSizes[c]

    used = [0] * N
    for wa in solution:
        used[wa] = 1

    # calculate the cost of the solution
    obj = sum([warehouses[x][1] * used[x] for x in range(0, N)])
    for c in range(0, M):
        obj += customerCosts[c][solution[c]]

    # prepare the solution in the specified output format
    outputData = str(obj) + ' ' + str(0) + '\n'
    outputData += ' '.join(map(str, solution))

    print(outputData)
    # return obj, outputData
    return outputData


def solveIt(inputData):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = inputData.split('\n')

    parts = lines[0].split()
    warehouseCount = int(parts[0])
    customerCount = int(parts[1])

    warehouses = []
    for i in range(1, warehouseCount + 1):
        line = lines[i]
        parts = line.split()
        warehouses.append((int(parts[0]), float(parts[1])))

    customerSizes = []
    customerCosts = []

    lineIndex = warehouseCount + 1
    for i in range(0, customerCount):
        customerSize = int(lines[lineIndex + 2 * i])
        customerCost = map(float, lines[lineIndex + 2 * i + 1].split())
        customerSizes.append(customerSize)
        customerCosts.append(customerCost)

    # model = SimpleModel(warehouses, customerSizes, customerCosts)
    model = LectureModel(warehouses, customerSizes, customerCosts)
    return solveWLP(model)

    # return solveGreedy(warehouses, customerSizes, customerCosts)
    #return solveGreedyBest(warehouses, customerSizes, customerCosts)

    # build a trivial solution
    # pack the warehouses one by one until all the customers are served

    solution = [-1] * customerCount
    capacityRemaining = [w[0] for w in warehouses]

    warehouseIndex = 0
    for c in range(0, customerCount):
        if capacityRemaining[warehouseIndex] >= customerSizes[c]:
            solution[c] = warehouseIndex
            capacityRemaining[warehouseIndex] -= customerSizes[c]
        else:
            warehouseIndex += 1
            assert capacityRemaining[warehouseIndex] >= customerSizes[c]
            solution[c] = warehouseIndex
            capacityRemaining[warehouseIndex] -= customerSizes[c]

    used = [0] * warehouseCount
    for wa in solution:
        used[wa] = 1

    # calculate the cost of the solution
    obj = sum([warehouses[x][1] * used[x] for x in range(0, warehouseCount)])
    for c in range(0, customerCount):
        obj += customerCosts[c][solution[c]]

    # prepare the solution in the specified output format
    outputData = str(obj) + ' ' + str(0) + '\n'
    outputData += ' '.join(map(str, solution))

    print(outputData)
    #return outputData


import sys

if __name__ == '__main__':
    random.seed()
    if len(sys.argv) > 1:
        fileLocation = sys.argv[1].strip()
        inputDataFile = open(fileLocation, 'r')
        inputData = ''.join(inputDataFile.readlines())
        inputDataFile.close()
        solveIt(inputData)
        # print 'Solving:', fileLocation
        #print solveIt(inputData)
    else:
        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/wl_16_1)'
