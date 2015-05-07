#!/usr/bin/python
# -*- coding: utf-8 -*-
import collections
import copy
import time
import sys

sys.setrecursionlimit(100000000)

colors = []
best_colors = []
baseline = 0

def solve_it(input_data):
    # tmpFile = open('2.txt', 'r')
    # out = tmpFile.read()
    # tmpFile.close()
    # return out
    global best_colors
    global baseline
    global colors

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    neighbors = [set() for i in range(node_count)]
    neighbor_colors = [collections.defaultdict(int) for i in range(node_count)]
    candidate_colors = set()
    colors = [0] * node_count
    best_colors = []
    baseline = node_count

    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))
        neighbors[int(parts[0])].add(int(parts[1]))
        neighbors[int(parts[1])].add(int(parts[0]))

    def get_neighbor_colors(vertex, ordered=True):
        neighbor_colors_v = set()
        for neighbor in neighbors[vertex]:
            if not ordered or neighbor < vertex:
                neighbor_colors_v.add(colors[neighbor])
        return neighbor_colors_v

    neighbors_num = [(i, len(neighbor)) for i, neighbor in enumerate(neighbors)]
    orders = sorted(neighbors_num, key=lambda num: num[1], reverse=True)
    # use dp
    def dfs(order_index, color, neighbor_colors, candidate_colors):
        global best_colors
        global baseline
        global colors

        if order_index >= node_count:  # won't happen internally
            return
        index = orders[order_index][0]
        candidate_colors.add(color)
        colors[index] = color

        for neighbor in neighbors[index]:
            neighbor_colors[neighbor][color] += 1
            # print 'neighbor %d of index %d add color %d:' % (neighbor, index, color)

        crnt_color_count = len(candidate_colors)
        if order_index == node_count - 1:
            if crnt_color_count < baseline:  # better
                print 'Baseline: %d -> %d' % (baseline, crnt_color_count)
                best_colors = copy.copy(colors)
                baseline = crnt_color_count
            return

        next_index = orders[order_index + 1][0]
        next_candidate_colors = candidate_colors.difference(neighbor_colors[next_index])
        new_color = len(candidate_colors)
        next_candidate_colors.add(new_color)

        for next_color in next_candidate_colors:
            if next_color == new_color and crnt_color_count >= baseline - 1:  # can't be better
                continue

            neighbors_ok = True
            for neighbor in neighbors[next_index]:
                if next_color not in neighbor_colors[neighbor] and len(neighbor_colors[neighbor]) >= baseline - 2:
                    neighbors_ok = False
                    break
            if not neighbors_ok:  # neighbors not ok (4x improvement)
                continue

            dfs(order_index + 1, next_color, neighbor_colors, candidate_colors)

            # clear next_index
            for neighbor in neighbors[next_index]:
                neighbor_colors[neighbor][next_color] -= 1
                if neighbor_colors[neighbor][next_color] == 0: del neighbor_colors[neighbor][next_color]
                # print 'neighbor %d of index %d remove color %d:' % (neighbor, next_index, next_color)
            if next_color == new_color:
                candidate_colors.remove(next_color)

    dfs(0, 0, neighbor_colors, candidate_colors)

    # check
    for v1, v2 in edges:
        if best_colors[v1] == best_colors[v2]:
            print 'Error: colors of %s = colors of %s.' % (v1, v2)

    # prepare the solution in the specified output format
    output_data = str(len(set(best_colors))) + ' ' + str(1) + '\n'
    output_data += ' '.join(map(str, best_colors))

    return output_data



if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        input_data_file = open(file_location, 'r')
        input_data = ''.join(input_data_file.readlines())
        input_data_file.close()
        time_start = time.time()
        print solve_it(input_data)
        time_end = time.time()
        print 'Time: %f s' % (time_end - time_start)
    else:
        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)'

