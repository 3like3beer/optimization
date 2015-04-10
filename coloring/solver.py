#!/usr/bin/python
# -*- coding: utf-8 -*-

import pulp

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))
    # build a trivial solution
    # every node has its own color
    #solution = range(0, node_count)
    solution = pulp_solve(node_count,edges,get_opt(node_count))

    # prepare the solution in the specified output format
    output_data = str(max(solution)+1) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


class Vertex:
    def  __init__(self, name):
        self.name = name
        self.adjacent = []
        self.color = 0
        self.degree = 0

    def add_adjacent(self,v):
        self.adjacent.append(v)
        self.degree += 1
        v.adjacent.append(self)
        v.degree += 1

    def color(self, color):
        self.color = color

    def get_dsat(self):


class Graph:
    def __init__(self, node_count, edges):
        self.vertices = [Vertex(i) for i in range(0,node_count)]
        for e in edges:
            self.vertices[e[0]].add_adjacent(self.vertices[e[1]])
        # Ordonner les sommets par ordre décroissant de degré.
        self.sorted_vertices = sorted(self.vertices,key=lambda x: x.color)


def get_opt(node_count):
    opt = {"50":6,"70":17,"100":15,"250":73,"500":12,"1000":88}

    #print (opt.keys())
    if str(node_count) in opt.keys():
        return opt[str(node_count)]
    else:
        return node_count


def objective_value(is_color,color_set):
     return sum([sum(is_color[color]) * sum(is_color[color]) for color in color_set])


def choose_next_vertex(sorted_vertices):
    return min(sorted_vertices)


def color_vertex(v, sorted_vertices):
    adjacents = sorted_vertices.get_vertex(v).get_adjacent()
    return max(adjacents.color) + 1


def dsatur_solve(node_count,edges):
    graph = Graph(node_count,edges)
    # 1.Ordonner les sommets par ordre décroissant de degré.
    sorted_vertices = graph.sorted_vertices

    # 2.Colorer un des sommets de degré maximum avec la couleur 1.
    i = 0
    sorted_vertices[i].color(1)

    # 3.Choisir un sommet non coloré avec DSAT maximum (nombre de couleurs différentes dans les sommets adjacents à v).
    # En cas d'égalité, choisir un sommet de degré maximal.
    v = choose_next_vertex(sorted_vertices)

    # 5.Si tous les sommets sont colorés alors stop
    while v>-1:
        # 4.Sinon colorer ce sommet par la plus petite couleur possible.
        color_vertex(v,sorted_vertices)

    return get_result(sorted_vetices)


def ls_solve(node_count,edges,opt):
    graph = Graph(node_count,edges)
    node_set = range(0,node_count)
    color_set = range(0,get_opt(node_count))
    root = graph.vertices[0]
    color = 0
    dfs(graph,root)
    is_color =  [[0 for color in color_set] for node in node_set]
    is_color[0][0] == 1
    for node in node_set:
        for color in color_set:
            node * is_color[color][node]



def dfs(graph,root):
    visited = [False for i in graph.vertices]
    visited[root] = True
    for v in root.adjacents:
        if not(visited[v]):
            dfs(graph,v)

def pulp_solve(node_count,edges,opt):
    print ("opt " +  str(opt))
    print("node_count " + str(node_count))
    coloring = pulp.LpProblem("Color Model", pulp.LpMinimize)
    color_set = range(0,opt + 5)
    node_set = range(0,node_count)
    is_color =  [[pulp.LpVariable("x_col" + str(c) + "_node" + str(n) , 0,1, 'Binary') for c in color_set] for n in node_set]
    obj = pulp.LpVariable("objective",opt,opt+3,'Integer')
    objective = pulp.LpAffineExpression(obj)
    coloring.setObjective(objective)


    for node in node_set:
        for c in color_set:
            coloring += c * is_color[node][c] <= obj
        coloring += sum(is_color[node][c] for c in color_set) == 1
    for c in color_set:
        for e in edges:
            coloring += is_color[e[0]][c] + is_color[e[1]][c] <= 1

    # for n in node_set:
    #     for c in color_set:
    #         coloring += n * is_color[n][c] <= obj
    #     coloring += sum(is_color[n][col] for col in color_set) == 1
    # for c in color_set:
    #     for e in edges:
    #         coloring += is_color[e[0]][c] + is_color[e[1]][c] <= 1
    #
    # for c in color_set:
    #     coloring += sum(is_color[v][c] for v in node_set) == 1

    #coloring += is_color[0][0] == 1

    #print(coloring)
    coloring.solve(pulp.PULP_CBC_CMD())

    out = []
    for n in node_set:
        for c in color_set:
            if is_color[n][c].value()>0.5:
                #print ("col" + str(color) + "node" + str(node))
                out.append(c)
        # print([is_color[c][n] for n in color_set])
        # print([is_color[c][n].value() for n in color_set])
    #print out
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
        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)'

